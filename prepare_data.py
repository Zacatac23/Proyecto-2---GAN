"""Audita, depura y divide los sprites antes de cualquier aumento de datos."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import zipfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
DUNGEON_CHARACTERS = set(range(84, 89)) | set(range(96, 101)) | set(range(108, 113)) | set(range(120, 125))
# Familias observadas en la rejilla: colores, tamaños o poses del mismo arquetipo.
CREATURE_FAMILIES = [
    [17,18,19], [21,22,23], [26,27,28,29,30], [41,42,43], [44,45],
    [51,52,53,54,55], [56,57,58,59,60], [61,62,63,64,65,71,72,73,74,75],
    [66,67], [80,81,82], [84,85], [86,87,88], [113,114,115,116,117],
    [131,132], [133,134], [135,136], [137,138], [139,140], [147,148],
    [164,165,166], [173,174,175],
]
DUNGEON_FAMILIES = [[85,86,88], [96,97,98], [99,100]]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_sprite(path):
    """Compone transparencia en negro y centra el contenido sin cambiar escala."""
    rgba = Image.open(path).convert('RGBA')
    bg = Image.new('RGBA', rgba.size, (0,0,0,255))
    bg.alpha_composite(rgba)
    rgb = np.asarray(bg.convert('RGB')).copy()
    mask = rgb.max(2) > 0
    if not mask.any():
        return Image.new('RGB', (16,16))
    yy, xx = np.where(mask)
    crop = Image.fromarray(rgb).crop((int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)))
    canvas = Image.new('RGB', (16,16))
    canvas.paste(crop, ((16-crop.width)//2,(16-crop.height)//2))
    return canvas


def montage(paths, destination, labels=None, columns=10, size=64):
    rows = (len(paths)+columns-1)//columns
    canvas=Image.new('RGB',(columns*(size+12)+12,rows*(size+28)+12),'white')
    draw=ImageDraw.Draw(canvas)
    for i,p in enumerate(paths):
        img=Image.open(p).convert('RGB').resize((size,size),Image.Resampling.NEAREST)
        x=12+(i%columns)*(size+12); y=12+(i//columns)*(size+28)
        canvas.paste(img,(x,y)); draw.text((x,y+size+3), labels[i] if labels else Path(p).stem,fill='black')
    canvas.save(destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip',type=Path)
    args=parser.parse_args()
    if args.zip:
        with zipfile.ZipFile(args.zip) as z:
            for item in z.infolist():
                target=(ROOT/item.filename).resolve()
                if not target.is_relative_to(ROOT/'data_raw'):
                    raise ValueError(f'Ruta inesperada en ZIP: {item.filename}')
                if not item.is_dir():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(z.read(item))
    files=sorted((ROOT/'data_raw').rglob('*.png'))
    if not files:
        raise SystemExit('Faltan datos: python prepare_data.py --zip RUTA/data_raw.zip')
    out=ROOT/'dataset'; out.mkdir(exist_ok=True)
    imgdir=out/'sprites'; imgdir.mkdir(exist_ok=True)
    rows=[]; images={}; content_seen={}
    for p in files:
        source='dungeon' if 'kenney_tinydungeon' in p.parts else 'creatures'
        index=int(p.stem.split('_')[-1]); key=f'{source}_{index:04d}'
        keep=(index in DUNGEON_CHARACTERS) if source=='dungeon' else index!=7
        reason='personaje o criatura revisado visualmente' if keep else (
            'escenario, objeto, interfaz, arma o armadura aislada' if source=='dungeon'
            else 'mano aislada: fragmento corporal sin personaje completo')
        row={'id':key,'source':source,'index':index,'raw_path':p.relative_to(ROOT).as_posix(),
             'raw_sha256':sha256(p),'keep':keep,'reason':reason,'duplicate_of':'','group':'','split':'','image_path':''}
        if keep:
            im=normalized_sprite(p); arr=np.asarray(im)
            content_hash=min(hashlib.sha256(arr.tobytes()).hexdigest(),hashlib.sha256(arr[:,::-1].tobytes()).hexdigest())
            if content_hash in content_seen:
                row.update(keep=False,reason='duplicado exacto o reflejo exacto',duplicate_of=content_seen[content_hash])
            else:
                content_seen[content_hash]=key; images[key]=im
                target=imgdir/f'{key}.png'
                im.resize((64,64),Image.Resampling.NEAREST).save(target)
                row['image_path']=target.relative_to(ROOT).as_posix()
        rows.append(row)
    accepted={r['id']:r for r in rows if r['keep']}
    parent={key:key for key in accepted}
    def find(key):
        while parent[key]!=key:
            parent[key]=parent[parent[key]]; key=parent[key]
        return key
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[max(ra,rb)]=min(ra,rb)
    for source,families in [('creatures',CREATURE_FAMILIES),('dungeon',DUNGEON_FAMILIES)]:
        for family in families:
            members=[f'{source}_{i:04d}' for i in family if f'{source}_{i:04d}' in accepted]
            for key in members[1:]: union(members[0],key)
    # Misma silueta, incluso reflejada: evita fuga de variantes cromaticas entre grupos.
    silhouettes={}
    for key,im in images.items():
        mask=(np.asarray(im).max(2)>0).astype(np.uint8)
        signature=min(mask.tobytes(),mask[:,::-1].tobytes())
        if signature in silhouettes: union(key,silhouettes[signature])
        else: silhouettes[signature]=key
    groups=sorted({find(key) for key in accepted})
    rng=random.Random(1729); rng.shuffle(groups)
    valid_groups=set(groups[:max(1,round(len(groups)*0.2))])
    for key,row in accepted.items():
        row['group']=find(key); row['split']='validation' if row['group'] in valid_groups else 'train'
    with (out/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/'manifest.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    selected=[r for r in rows if r['keep']]
    summary={'raw_count':len(rows),'accepted_count':len(selected),'rejected_count':len(rows)-len(selected),
             'duplicates_removed':sum(bool(r['duplicate_of']) for r in rows),
             'group_count':len(groups),'split_seed':1729,'validation_group_fraction':0.2,
             'train_count':sum(r['split']=='train' for r in selected),
             'validation_count':sum(r['split']=='validation' for r in selected),
             'train_groups':len(groups)-len(valid_groups),'validation_groups':len(valid_groups),
             'augmentation':'original y reflejo horizontal solo en entrenamiento, sin permutar RGB',
             'preprocessing':'negro, centrado de caja no negra en 16x16, NEAREST a 64x64',
             'sources':[{'name':'Tiny Dungeon','author':'Kenney','license':'CC0','url':'https://kenney.nl/assets/tiny-dungeon'},
                        {'name':'Tiny Creatures','author':'Clint Bellanger, con colaboracion de Kenney','license':'CC0','url':'https://opengameart.org/content/tiny-creatures'}]}
    summary['augmented_train_count']=summary['train_count']*2
    summary['manifest_sha256']=sha256(out/'manifest.csv')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    palette=sorted({tuple(pixel) for r in selected if r['split']=='train'
                    for pixel in np.asarray(images[r['id']]).reshape(-1,3).tolist()})
    (out/'palette.json').write_text(json.dumps(palette,indent=2),encoding='utf-8')
    for split in ['train','validation']:
        subset=[r for r in selected if r['split']==split]
        montage([ROOT/r['image_path'] for r in subset],out/f'{split}_grid.png',
                [r['id'].replace('creatures','c').replace('dungeon','d') for r in subset])
    assert {r['group'] for r in selected if r['split']=='train'}.isdisjoint({r['group'] for r in selected if r['split']=='validation'})
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__': main()
