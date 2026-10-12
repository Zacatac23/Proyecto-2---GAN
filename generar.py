"""Genera nuevos personajes o verifica la galeria final sin reentrenar."""
import argparse
from pathlib import Path

import torch

from gan_pipeline import ROOT, RUN, grid, image_from_tensor, load_generator, load_json, sample, verify_gallery


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verificar',action='store_true')
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--cantidad',type=int,default=16)
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--salida',type=Path,default=ROOT/'output_eval/nuevos')
    args=parser.parse_args()
    run=args.run.resolve()
    if args.verificar:
        verify_gallery(run); return
    if not 1<=args.cantidad<=200: parser.error('--cantidad debe estar entre 1 y 200')
    manifest=load_json(run/'galeria/manifest.json')
    device=torch.device('cpu')
    torch.set_num_threads(6)
    model,_=load_generator(ROOT/manifest['checkpoint'],device)
    seeds=list(range(args.seed,args.seed+args.cantidad))
    z=torch.cat([torch.randn(1,100,1,1,generator=torch.Generator().manual_seed(seed)) for seed in seeds])
    images=sample(model,z,device,batch=1)
    out=args.salida.resolve(); out.mkdir(parents=True,exist_ok=True)
    for index,seed in enumerate(seeds):
        image_from_tensor(images[index]).save(out/f'personaje_seed_{seed}.png')
        torch.save(z[index:index+1],out/f'personaje_seed_{seed}_z.pt')
    grid(images,out/'mosaico.png',[f'seed {seed}' for seed in seeds],columns=4,scale=96)
    print(f'Modelo {manifest["model"]}, epoca {manifest["epoch"]}; dispositivo {device}; salida {out}')


if __name__=='__main__': main()
