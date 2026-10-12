"""Figuras del informe, extraidas de los pesos, datos e historiales ejecutados."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch
from PIL import Image
from gan_pipeline import ROOT, RUN, load_json, grid, image_from_tensor, load_generator, sample


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=RUN)
    p.add_argument('--epoch',type=int,default=2000)
    a=p.parse_args(); run=a.run.resolve(); epoch=a.epoch
    out=run/'figures'; out.mkdir(exist_ok=True)
    tags=['base','hinge','hinge_sn']; labels={'base':'BCE','hinge':'Hinge','hinge_sn':'Hinge + SN'}
    gallery=load_json(run/'galeria/manifest.json'); tag=gallery['model']
    fixed=torch.load(run/'fixed_noise.pt',weights_only=True)
    models={t:load_generator(run/t/f'generator_epoch_{epoch:04d}.pt',torch.device('cpu'))[0] for t in tags}
    fixed_images={t:sample(models[t],fixed[:8],torch.device('cpu')) for t in tags}
    evolution=Image.new('RGB',(646,4*168),'white')
    from PIL import ImageDraw,ImageFont
    draw=ImageDraw.Draw(evolution)
    label_font=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',22)
    for row,e in enumerate([1,100,500,epoch]):
        frame=Image.open(run/tag/f'fixed_epoch_{e:04d}.png')
        draw.multiline_text((5,row*168+50),f'Ep.\n{e}',font=label_font,fill='black')
        evolution.paste(frame,(74,row*168+11))
    evolution.save(out/'fixed_evolution_selected.png')
    for name,pair in [('loss_comparison',['base','hinge']),('stabilization_comparison',['hinge','hinge_sn']),('model_comparison',tags)]:
        fig,axes=plt.subplots(len(pair),8,figsize=(10,len(pair)*1.5))
        for row,t in enumerate(pair):
            for col,img in enumerate(fixed_images[t]):
                ax=axes[row,col]; ax.imshow(image_from_tensor(img),interpolation='nearest'); ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values(): spine.set_visible(False)
                if col==0: ax.set_ylabel(labels[t],fontsize=17,color='black')
        fig.tight_layout(pad=.6); fig.savefig(out/f'{name}.png',dpi=180); plt.close(fig)
    plt.rcParams.update({'font.size':15})
    fig,axes=plt.subplots(3,1,figsize=(8,5.5),sharex=True)
    for ax,t in zip(axes,tags):
        hist=load_json(run/t/'history.json'); x=[r['epoch'] for r in hist]
        ax.plot(x,[r['D'] for r in hist],color='black',lw=.7,label='D')
        ax.plot(x,[r['G'] for r in hist],color='black',lw=.7,ls='--',label='G')
        ax.set_ylabel('Pérdida'); ax.set_title(labels[t],fontsize=15);ax.legend(loc='upper left',fontsize=15,frameon=False,ncol=2)
        ax.spines[['top','right']].set_visible(False)
    axes[-1].set_xlabel('Época');fig.tight_layout();fig.savefig(out/'all_losses.png',dpi=200);plt.close(fig)
    neighbors=torch.load(run/tag/f'neighbors_epoch_{epoch:04d}.pt',weights_only=True)
    candidates=torch.load(run/tag/f'candidates_epoch_{epoch:04d}.pt',weights_only=True)
    for row in gallery['characters']:
        index=row['candidate_index']; nearest=neighbors['train_refs'][neighbors['rms_idx'][index]]
        image_from_tensor(nearest).save(out/f"neighbor_{row['number']:02d}.png")
    # Caso mas similar segun ResNet entre los diez elegidos, con ambos vecinos.
    critical=max(gallery['characters'],key=lambda r:float(neighbors['cosine'][r['candidate_index']]))
    i=critical['candidate_index']
    grid(torch.stack([candidates['images'][i],neighbors['train_refs'][neighbors['rms_idx'][i]],neighbors['train_refs'][neighbors['resnet_idx'][i]]]),out/'critical_case.png',
         [f"Generado {critical['number']:02d}",f"RMS {float(neighbors['rms'][i]):.3f}",f"Coseno {float(neighbors['cosine'][i]):.3f}"],columns=3,scale=128)
    (out/'critical_case.json').write_text(json.dumps(critical,indent=2),encoding='utf-8')
    # Diagrama del ciclo de entrenamiento y uso del modelo congelado.
    fig,ax=plt.subplots(figsize=(8,2.8)); ax.set_xlim(0,10);ax.set_ylim(0,4);ax.axis('off')
    def box(x,y,w,h,text):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',facecolor='white',edgecolor='black',linewidth=1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12,color='black')
    def arrow(start,end): ax.annotate('',end,start,arrowprops={'arrowstyle':'->','color':'black','lw':1})
    box(.15,2.85,2,.8,'312 sprites\nFiltrar/agrupar')
    box(2.65,2.85,2,.8,'Separar 164/35\nAumentar train')
    box(5.15,2.85,2,.8,'Reales + G(z)\nD: real/falsa')
    box(7.65,2.85,2,.8,'Pérdidas D/G\nAdam D y G')
    arrow((2.21,3.25),(2.59,3.25));arrow((4.71,3.25),(5.09,3.25));arrow((7.21,3.25),(7.59,3.25))
    arrow((8.65,2.79),(6.15,2.79));ax.text(7.4,2.45,'Repetir por lote y época',ha='center',fontsize=13)
    box(.15,.9,2,.8,'Ruido fijo\nCurvas/vecinos')
    box(2.65,.9,2,.8,'Elegir modelo\nPesos y z')
    box(5.15,.9,2,.8,'Nueva semilla: z\nG en eval')
    box(7.65,.9,2,.8,'Paleta 16 × 16\nPNG 64 × 64')
    arrow((2.21,1.3),(2.59,1.3));arrow((4.71,1.3),(5.09,1.3));arrow((7.21,1.3),(7.59,1.3))
    ax.plot([9.72,9.95,9.95,1.15,1.15],[3.25,3.25,2.15,2.15,1.95],color='black',lw=1)
    arrow((1.15,1.95),(1.15,1.76));ax.text(5,.3,'Al generar imágenes nuevas no se actualizan los pesos.',ha='center',fontsize=13)
    fig.tight_layout();fig.savefig(out/'flujo.png',dpi=200);plt.close(fig)
    print('Figuras:',out)


if __name__=='__main__': main()
