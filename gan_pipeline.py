"""DCGAN reproducible: entrenamiento controlado, evaluacion y regeneracion.

La red aprende la rejilla nativa 16x16 y exporta 64x64 mediante NEAREST.
Los datos de validacion nunca participan en los gradientes ni en el aumento.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/paleta_2000'
VARIANTS={'base':{'loss':'bce','spectral_norm':False},
          'hinge':{'loss':'hinge','spectral_norm':False},
          'hinge_sn':{'loss':'hinge','spectral_norm':True}}


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')


def load_json(path): return json.loads(path.read_text(encoding='utf-8'))


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def state_hash(module):
    digest=hashlib.sha256()
    for key,value in module.state_dict().items():
        digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Generator(nn.Module):
    def __init__(self,z_dim=100,width=64,palette=None):
        super().__init__()
        self.use_palette=palette is not None
        if self.use_palette:
            self.register_buffer('palette',torch.tensor(palette,dtype=torch.float32)/127.5-1)
        output_channels=len(palette) if self.use_palette else 3
        self.net=nn.Sequential(
            nn.ConvTranspose2d(z_dim,width*4,4,1,0,bias=False),nn.BatchNorm2d(width*4),nn.ReLU(True),
            nn.ConvTranspose2d(width*4,width*2,4,2,1,bias=False),nn.BatchNorm2d(width*2),nn.ReLU(True),
            nn.ConvTranspose2d(width*2,output_channels,4,2,1,bias=False),nn.Identity() if self.use_palette else nn.Tanh())
    def forward(self,z):
        native=self.net(z)
        if self.use_palette:
            probabilities=native.softmax(1)
            hard=F.one_hot(probabilities.argmax(1),len(self.palette)).permute(0,3,1,2).to(probabilities.dtype)
            # Forward discreto, gradiente sustituto de softmax en entrenamiento.
            assignment=hard-probabilities.detach()+probabilities if self.training else hard
            native=torch.einsum('bkhw,kc->bchw',assignment,self.palette)
        return F.interpolate(native,size=(64,64),mode='nearest')


class Discriminator(nn.Module):
    def __init__(self,width=64):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,width,4,2,1,bias=False),nn.LeakyReLU(0.2,True),
            nn.Conv2d(width,width*2,4,2,1,bias=False),nn.BatchNorm2d(width*2),nn.LeakyReLU(0.2,True),
            nn.Conv2d(width*2,1,4,1,0,bias=False))
    def forward(self,x):
        return self.net(F.avg_pool2d(x,4)).flatten()


def initialize(module):
    if isinstance(module,(nn.Conv2d,nn.ConvTranspose2d)):
        nn.init.normal_(module.weight,0,0.02)
    elif isinstance(module,nn.BatchNorm2d):
        nn.init.normal_(module.weight,1,0.02); nn.init.zeros_(module.bias)


def read_dataset():
    manifest=load_json(ROOT/'dataset/manifest.json')
    accepted=[row for row in manifest if row['keep']]
    def images(rows):
        return torch.stack([torch.from_numpy(np.array(Image.open(ROOT/row['image_path']).convert('RGB')).copy()).permute(2,0,1).float()/255 for row in rows])
    train_rows=[row for row in accepted if row['split']=='train']
    val_rows=[row for row in accepted if row['split']=='validation']
    return images(train_rows),images(val_rows),train_rows,val_rows


def image_from_tensor(x):
    return Image.fromarray((x.detach().cpu().permute(1,2,0).numpy()*255+0.5).clip(0,255).astype(np.uint8))


def grid(x,path,labels=None,columns=8,scale=64):
    count=len(x); rows=(count+columns-1)//columns
    labelheight=20 if labels else 0
    canvas=Image.new('RGB',(columns*(scale+6)+6,rows*(scale+6+labelheight)+6),'white')
    draw=ImageDraw.Draw(canvas)
    for i,img in enumerate(x):
        left=6+(i%columns)*(scale+6); top=6+(i//columns)*(scale+6+labelheight)
        canvas.paste(image_from_tensor(img).resize((scale,scale),Image.Resampling.NEAREST),(left,top))
        if labels: draw.text((left,top+scale+2),str(labels[i]),fill='black')
    path.parent.mkdir(parents=True,exist_ok=True); canvas.save(path)


@torch.no_grad()
def recalibrate_bn(model,device,z_dim=100,batches=20,seed=12345):
    """Usa un RNG independiente y no modifica el orden ni el ruido de entrenamiento."""
    old={}
    for name,module in model.named_modules():
        if isinstance(module,nn.BatchNorm2d):
            old[name]=module.momentum; module.reset_running_stats(); module.momentum=None
    model.train()
    rng=torch.Generator(device=device).manual_seed(seed)
    for _ in range(batches): model(torch.randn(128,z_dim,1,1,device=device,generator=rng))
    model.eval()
    for name,module in model.named_modules():
        if name in old: module.momentum=old[name]


@torch.no_grad()
def sample(model,z,device,batch=64):
    model.eval()
    return torch.cat([(model(part.to(device)).cpu()*0.5+0.5).clamp(0,1) for part in z.split(batch)])


def diversity(x):
    flat=F.avg_pool2d(x,4).flatten(1)
    distance=torch.cdist(flat,flat)/flat.shape[1]**0.5
    return float(distance.sum()/(len(x)*(len(x)-1))) if len(x)>1 else 0.0


def pixel_nearest(generated,references):
    g=F.avg_pool2d(generated,4).flatten(1); r=F.avg_pool2d(references,4).flatten(1)
    distances=torch.cdist(g,r)/g.shape[1]**0.5
    return distances.min(1)


def train(tag,epochs,seed,width=64,batch_size=32,run=RUN,use_palette=False,learning_rate_d=2e-4,discriminator_width=64):
    seed_everything(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output=run/tag
    if (output/'history.json').exists():
        raise SystemExit(f'La corrida ya existe: {output}. Use --run para otra corrida.')
    output.mkdir(parents=True,exist_ok=True)
    train01,valid01,train_rows,val_rows=read_dataset()
    data=torch.cat([train01,train01.flip(3)]).to(device)*2-1
    palette=load_json(ROOT/'dataset/palette.json') if use_palette else None
    model=Generator(width=width,palette=palette).to(device); discriminator=Discriminator(width=discriminator_width).to(device)
    model.apply(initialize); discriminator.apply(initialize)
    initial_g,initial_d=state_hash(model),state_hash(discriminator)
    option=VARIANTS[tag]
    if option['spectral_norm']:
        for module in discriminator.modules():
            if isinstance(module,nn.Conv2d): spectral_norm(module)
    optimizer_g=torch.optim.Adam(model.parameters(),lr=2e-4,betas=(0.5,0.999))
    optimizer_d=torch.optim.Adam(discriminator.parameters(),lr=learning_rate_d,betas=(0.5,0.999))
    # Verifica que el optimizador realmente referencia la red que se ejecuta.
    assert {id(p) for p in model.parameters()}=={id(p) for group in optimizer_g.param_groups for p in group['params']}
    fixed=torch.randn(16,100,1,1,generator=torch.Generator().manual_seed(123))
    probes=torch.randn(128,100,1,1,generator=torch.Generator().manual_seed(777))
    torch.save(fixed,run/'fixed_noise.pt')
    metadata={'tag':tag,**option,'epochs':epochs,'seed':seed,'latent_dim':100,'width':width,
       'batch_size':batch_size,'learning_rate':2e-4,'learning_rate_D':learning_rate_d,'discriminator_width':discriminator_width,'betas':[0.5,0.999],
       'native_resolution':16,'export_resolution':64,'upsampling':'nearest',
       'palette':palette,'palette_training':'hard argmax con gradiente sustituto softmax' if palette else None,
       'discriminator_batchnorm':True,'initial_G_sha256':initial_g,'initial_D_sha256':initial_d,
       'batch_order_seed':'4200 + epoch','training_noise_seed':'9000 + epoch','fixed_noise_seed':123,
       'evaluation_noise_seed':777,'bn_calibration_seed':12345,'bn_calibration_batches':20,
       'train_images':len(train01),'augmented_images':len(data),'validation_images':len(valid01),
       'manifest_sha256':load_json(ROOT/'dataset/summary.json')['manifest_sha256'],
       'device':str(device),'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,
       'python':platform.python_version(),'torch':str(torch.__version__),
       'generator_parameters':sum(p.numel() for p in model.parameters()),
       'discriminator_parameters':sum(p.numel() for p in discriminator.parameters()),
       'started_at':datetime.now().astimezone().isoformat()}
    write_json(output/'config.json',metadata)
    (output/'hypothesis.txt').write_text({
       'base':'Control: BCE no saturante, BatchNorm y sin normalizacion espectral.',
       'hinge':'Creemos que Hinge mejorara la estabilidad porque utiliza un margen en D y una perdida lineal en G. Lo consideraremos util si preserva diversidad y produce siluetas mas claras con el mismo ruido fijo, sin aumentar similitud excesiva con entrenamiento.',
       'hinge_sn':'Creemos que normalizacion espectral mejorara porque limita la amplificacion de cada convolucion de D. Lo consideraremos util si mejora las muestras y evita dominancia prolongada de D frente al control Hinge. Conservamos BatchNorm y todos los demas factores.'
    }[tag],encoding='utf-8')
    bce=nn.BCEWithLogitsLoss(); history=[]; snapshots={}; started=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train(); discriminator.train(); sums=np.zeros(4); steps=0
        order=torch.randperm(len(data),generator=torch.Generator().manual_seed(4200+epoch)).to(device)
        noise_rng=torch.Generator(device=device).manual_seed(9000+epoch)
        for indices in order.split(batch_size):
            if len(indices)<2: continue
            real=data[indices]; batch=len(real)
            for p in discriminator.parameters(): p.requires_grad_(True)
            optimizer_d.zero_grad(set_to_none=True)
            z=torch.randn(batch,100,1,1,device=device,generator=noise_rng)
            fake=model(z)
            out_real=discriminator(real); out_fake=discriminator(fake.detach())
            loss_d=(bce(out_real,torch.ones_like(out_real))+bce(out_fake,torch.zeros_like(out_fake))) if option['loss']=='bce' else (F.relu(1-out_real).mean()+F.relu(1+out_fake).mean())
            loss_d.backward(); optimizer_d.step()
            for p in discriminator.parameters(): p.requires_grad_(False)
            optimizer_g.zero_grad(set_to_none=True)
            out_g=discriminator(fake)
            loss_g=bce(out_g,torch.ones_like(out_g)) if option['loss']=='bce' else -out_g.mean()
            if not (torch.isfinite(loss_g) and torch.isfinite(loss_d)):
                raise RuntimeError(f'Perdida no finita en {tag} epoca {epoch}')
            loss_g.backward(); optimizer_g.step()
            sums+=np.array([loss_d.item(),loss_g.item(),out_real.detach().mean().item(),out_fake.detach().mean().item()]); steps+=1
        record={'epoch':epoch,'D':float(sums[0]/steps),'G':float(sums[1]/steps),
                'real_score':float(sums[2]/steps),'fake_score':float(sums[3]/steps),
                'elapsed_seconds':time.perf_counter()-started}
        if epoch==1 or epoch%25==0 or epoch==epochs:
            evaluated=copy.deepcopy(model); recalibrate_bn(evaluated,device)
            fixed_images=sample(evaluated,fixed,device); snapshots[epoch]=fixed_images
            generated=sample(evaluated,probes,device)
            dtrain,_=pixel_nearest(generated,train01); dval,_=pixel_nearest(generated,valid01)
            record.update(diversity=diversity(generated),nearest_train_rms_mean=float(dtrain.mean()),
                          nearest_validation_rms_mean=float(dval.mean()),
                          near_copies_rms_002=int((dtrain<0.02).sum()),
                          foreground_fraction=float((F.avg_pool2d(generated,4).max(1).values>0.1).float().mean()))
            grid(fixed_images,output/f'fixed_epoch_{epoch:04d}.png',columns=8)
            torch.save({'G':{key:value.cpu() for key,value in evaluated.state_dict().items()},
                        'config':metadata,'epoch':epoch},output/f'generator_epoch_{epoch:04d}.pt')
            del evaluated
        history.append(record); write_json(output/'history.json',history)
        if epoch==1 or epoch%25==0:
            print(f"{tag} epoch={epoch}/{epochs} D={record['D']:.4f} G={record['G']:.4f} seconds={record['elapsed_seconds']:.1f}"+
                  (f" diversity={record['diversity']:.4f} near_copies={record['near_copies_rms_002']}" if 'diversity' in record else ''),flush=True)
    metadata['finished_at']=datetime.now().astimezone().isoformat(); metadata['duration_seconds']=time.perf_counter()-started
    write_json(output/'config.json',metadata)
    # Una figura con los mismos vectores durante todo el entrenamiento.
    chosen=[e for e in [1,25,50,100,300,500,750,1000,1250,1500] if e in snapshots]
    if epochs not in chosen: chosen.append(epochs)
    grid(torch.cat([snapshots[e] for e in chosen]),output/'fixed_evolution.png',
         [f'ep {e} z {j}' for e in chosen for j in range(16)],columns=16,scale=48)
    plot_history(history,output/'loss_curves.png',tag)
    print(f'FIN {tag}: {metadata["duration_seconds"]:.1f}s',flush=True)


def plot_history(history,path,tag):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(9,3.3))
    epochs=[r['epoch'] for r in history]
    for key,style in [('D','-'),('G','--')]: axes[0].plot(epochs,[r[key] for r in history],style,color='black',label=key)
    for key,style,label in [('real_score','-','Real'),('fake_score','--','Generado')]:
        axes[1].plot(epochs,[r[key] for r in history],style,color='black',label=label)
    axes[0].set_title(f'Perdidas: {tag}'); axes[0].set_ylabel('Perdida')
    axes[1].set_title('Puntuacion del discriminador'); axes[1].set_ylabel('Logit / puntuacion')
    for ax in axes: ax.set_xlabel('Epoca'); ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path,dpi=170); plt.close(fig)


def load_generator(path,device):
    # Compatibilidad con las primeras corridas locales, que guardaron TorchVersion.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload=torch.load(path,map_location='cpu',weights_only=True)
    model=Generator(width=payload['config']['width'],palette=payload['config'].get('palette')).to(device)
    model.load_state_dict(payload['G']); model.eval()
    return model,payload


def evaluate(tag,epoch,run=RUN,cpu=False):
    seed_everything(42)
    device=torch.device('cpu' if cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    output=run/tag; model,payload=load_generator(output/f'generator_epoch_{epoch:04d}.pt',device)
    train01,valid01,train_rows,val_rows=read_dataset()
    z=torch.cat([torch.randn(1,100,1,1,generator=torch.Generator().manual_seed(1000+i)) for i in range(200)])
    generated=sample(model,z,device,batch=1 if cpu else 64)
    torch.save({'images':generated,'z':z,'epoch':epoch,'tag':tag,'device':str(device),'batch':1 if cpu else 64},output/f'candidates_epoch_{epoch:04d}.pt')
    grid(generated,output/f'candidates_epoch_{epoch:04d}.png',list(range(200)),columns=10,scale=64)
    distance,idx=pixel_nearest(generated,torch.cat([train01,train01.flip(3)]))
    valdist,validx=pixel_nearest(generated,valid01)
    # Caracteristicas ImageNet: evidencia auxiliar, no prueba de originalidad.
    from torchvision import models
    feature_model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
    feature_model.fc=nn.Identity()
    @torch.no_grad()
    def features(x):
        result=[]
        for chunk in x.split(32):
            resized=F.interpolate(chunk.to(device),size=(224,224),mode='nearest')
            normalized=(resized-torch.tensor([.485,.456,.406],device=device).view(1,3,1,1))/torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
            result.append(F.normalize(feature_model(normalized),dim=1).cpu())
        return torch.cat(result)
    real_refs=torch.cat([train01,train01.flip(3)])
    realfeatures=features(real_refs); genfeatures=features(generated)
    sims,simidx=(genfeatures@realfeatures.T).max(1)
    original_features=realfeatures[:len(train01)]
    original_sims=original_features@original_features.T
    # Referencia entre grupos diferentes, no otras variantes de la misma familia.
    for i,r in enumerate(train_rows):
        for j,s in enumerate(train_rows):
            if r['group']==s['group']: original_sims[i,j]=-1
    reference=original_sims.max(1).values
    fggen=(F.avg_pool2d(generated,4).max(1).values>0.1).flatten(1).float()
    fgrefs=(F.avg_pool2d(real_refs,4).max(1).values>0.1).flatten(1).float()
    intersection=fggen@fgrefs.T
    union=fggen.sum(1)[:,None]+fgrefs.sum(1)[None,:]-intersection
    iou,iouidx=(intersection/union.clamp_min(1)).max(1)
    metrics={'tag':tag,'epoch':epoch,'samples':200,'seeds':'1000 + index','generation_device':str(device),'generation_batch':1 if cpu else 64,
      'diversity':diversity(generated),'real_train_diversity':diversity(train01),
      'real_validation_diversity':diversity(valid01),'nearest_train_rms_mean':float(distance.mean()),
      'nearest_validation_rms_mean':float(valdist.mean()),'near_copies_rms_002':int((distance<.02).sum()),
      'resnet_cosine_mean':float(sims.mean()),'resnet_cosine_max':float(sims.max()),
      'real_other_group_cosine_median':float(reference.median()),
      'foreground_fraction':float(fggen.mean()),'nearest_silhouette_iou_mean':float(iou.mean()),
      'limitations':['RMS favorece fondos negros y depende de alineacion.',
        'ResNet-18 se entreno en fotografias, no en pixel art.',
        'Umbral RMS 0.02 es una alerta heuristica, no prueba de ausencia de memorizacion.',
        'Validacion por familias reduce fuga conocida, pero no demuestra independencia semantica completa.',
        'Una corrida por condicion no permite establecer significacion estadistica.'],
      'per_candidate':[{'index':i,'seed':1000+i,'rms_train':float(distance[i]),
          'nearest_train_id':train_rows[int(idx[i])%len(train_rows)]['id'],
          'nearest_train_flip':bool(idx[i]>=len(train_rows)),
          'rms_validation':float(valdist[i]),'resnet_cosine':float(sims[i]),
          'nearest_resnet_id':train_rows[int(simidx[i])%len(train_rows)]['id'],
          'nearest_resnet_flip':bool(simidx[i]>=len(train_rows)),
          'silhouette_iou':float(iou[i])} for i in range(200)]}
    write_json(output/f'evaluation_epoch_{epoch:04d}.json',metrics)
    torch.save({'train_refs':real_refs,'rms_idx':idx,'resnet_idx':simidx,'cosine':sims,
                'rms':distance,'train_rows':train_rows},output/f'neighbors_epoch_{epoch:04d}.pt')
    print(json.dumps({k:v for k,v in metrics.items() if k not in ['per_candidate','limitations']},indent=2),flush=True)


def export_gallery(tag,epoch,indices,run=RUN):
    if len(indices)!=10 or len(set(indices))!=10 or not all(0<=i<200 for i in indices):
        raise ValueError('Se requieren 10 indices distintos entre 0 y 199')
    source=run/tag
    payload=torch.load(source/f'candidates_epoch_{epoch:04d}.pt',weights_only=True)
    generated,z=payload['images'],payload['z']
    modelpath=source/f'generator_epoch_{epoch:04d}.pt'
    directory=run/'galeria'; directory.mkdir(exist_ok=True)
    names=['Morvath','Ignis','Grog','Kaelen','Sombra','Zul Karr','Glacius','Mossgolem','Aveline','Ursok']
    rows=[]
    for pos,index in enumerate(indices,1):
        name=f'personaje_{pos:02d}'
        image_from_tensor(generated[index]).save(directory/f'{name}.png')
        torch.save(z[index:index+1],directory/f'{name}_z.pt')
        rows.append({'number':pos,'name':names[pos-1],'candidate_index':index,'seed':1000+index,'image':f'{name}.png','z':f'{name}_z.pt'})
    gallery={'model':tag,'epoch':epoch,'checkpoint':modelpath.relative_to(ROOT).as_posix(),
             'checkpoint_sha256':hashlib.sha256(modelpath.read_bytes()).hexdigest(),
             'candidate_count':200,'selected_count':10,'selection_rate':0.05,
             'criterion':'Revision visual de 200 candidatos: legibilidad relativa, menor dispersion de pixeles y variedad de forma y color. Persisten deformaciones y repeticion de arquetipos; los vecinos no prueban originalidad semantica.',
             'generation_device':payload.get('device','cuda'),'generation_batch':payload.get('batch',64),
             'selection_scope':'La tasa 10/200 corresponde al conjunto final. Se inspeccionaron ademas rejillas de evaluacion de las tres condiciones y los checkpoints.',
             'characters':rows}
    write_json(directory/'manifest.json',gallery)
    (directory/'semillas.txt').write_text('\n'.join(f"personaje_{r['number']:02d}\tindice={r['candidate_index']}\tseed={r['seed']}" for r in rows)+'\n',encoding='utf-8')
    grid(generated[indices],directory/'galeria_mosaico.png',[f"{r['number']:02d} seed {r['seed']}" for r in rows],columns=5,scale=96)
    neighbors=torch.load(source/f'neighbors_epoch_{epoch:04d}.pt',weights_only=True)
    for mode,label in [('rms_idx','RMS'),('resnet_idx','ResNet')]:
        paired=[]; labels=[]
        for pos,index in enumerate(indices,1):
            paired.extend([generated[index],neighbors['train_refs'][neighbors[mode][index]]])
            metric=neighbors['rms'][index] if mode=='rms_idx' else neighbors['cosine'][index]
            labels.extend([f'Generado {pos:02d}',f'{label} {float(metric):.3f}'])
        grid(torch.stack(paired),directory/f'vecinos_{label.lower()}.png',labels,columns=4,scale=96)
    verify_gallery(run)


def verify_gallery(run=RUN):
    directory=run/'galeria'; manifest=load_json(directory/'manifest.json')
    checkpoint=ROOT/manifest['checkpoint']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==manifest['checkpoint_sha256']
    # CPU deliberadamente: demuestra que no se necesita la GPU para regenerar.
    model,_=load_generator(checkpoint,torch.device('cpu'))
    records=[]
    for row in manifest['characters']:
        z=torch.load(directory/row['z'],map_location='cpu',weights_only=True)
        generated=sample(model,z,torch.device('cpu'))[0]
        original=torch.from_numpy(np.array(Image.open(directory/row['image']).convert('RGB')).copy()).permute(2,0,1).float()/255
        error=float((generated-original).abs().max())
        seed_z=torch.randn(1,100,1,1,generator=torch.Generator().manual_seed(row['seed']))
        records.append({'character':row['number'],'max_error':error,'z_matches_seed':bool(torch.equal(z,seed_z)),'passed':error<2/255})
    result={'passed':all(r['passed'] and r['z_matches_seed'] for r in records),'device':'cpu','tolerance':2/255,'characters':records}
    write_json(directory/'verification.json',result)
    print('REGENERABILIDAD:',json.dumps(result),flush=True)
    if not result['passed']: raise RuntimeError('La galeria no se pudo regenerar')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','evaluate','export','verify'])
    parser.add_argument('--tag',choices=list(VARIANTS),default='base')
    parser.add_argument('--epochs',type=int,default=2000)
    parser.add_argument('--epoch',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--palette',action='store_true',help='Salida categorica de la paleta del entrenamiento')
    parser.add_argument('--lr-d',type=float,default=2e-4,help='Tasa de aprendizaje de D; G conserva 0.0002')
    parser.add_argument('--d-width',type=int,default=64,help='Canales base de D; G conserva 64')
    parser.add_argument('--cpu',action='store_true',help='Evaluacion canonica en CPU, una muestra por pase')
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--indices',type=int,nargs='+')
    args=parser.parse_args(); run=args.run.resolve(); run.mkdir(parents=True,exist_ok=True)
    if args.command=='train': train(args.tag,args.epochs,args.seed,run=run,use_palette=args.palette,learning_rate_d=args.lr_d,discriminator_width=args.d_width)
    elif args.command=='evaluate': evaluate(args.tag,args.epoch,run,args.cpu)
    elif args.command=='export': export_gallery(args.tag,args.epoch,args.indices,run)
    else: verify_gallery(run)


if __name__=='__main__': main()
