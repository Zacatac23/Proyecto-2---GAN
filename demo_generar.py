"""Genera personajes con los pesos existentes, sin ejecutar el entrenamiento.

Uso: python demo_generar.py --seed 2026 --cantidad 16
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn


class Generator(nn.Module):
    """La misma arquitectura y nombres de capas que en proyecto-2.ipynb."""

    def __init__(self):
        super().__init__()
        layers = []
        for index, (inputs, outputs) in enumerate(
            [(100, 512), (512, 256), (256, 128), (128, 64), (64, 3)]
        ):
            layers.append(nn.ConvTranspose2d(
                inputs, outputs, 4, 1 if index == 0 else 2,
                0 if index == 0 else 1, bias=False,
            ))
            if index < 4:
                layers.extend([nn.BatchNorm2d(outputs), nn.ReLU(True)])
            else:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


def to_image(tensor):
    pixels = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray((pixels * 255 + 0.5).clip(0, 255).astype(np.uint8))


def save_grid(images, labels, path, columns=5):
    scale, margin, caption = 192, 12, 24
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new('RGB', (columns * (scale + margin) + margin,
                              rows * (scale + margin + caption) + margin), '#20242c')
    draw = ImageDraw.Draw(canvas)
    for index, (img, label) in enumerate(zip(images, labels)):
        x = margin + (index % columns) * (scale + margin)
        y = margin + (index // columns) * (scale + margin + caption)
        canvas.paste(img.resize((scale, scale), Image.Resampling.NEAREST), (x, y))
        draw.text((x, y + scale + 5), label, fill='white')
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--cantidad', type=int, default=16)
    parser.add_argument('--salida', type=Path, default=Path('output_eval/demo'))
    args = parser.parse_args()
    if not 1 <= args.cantidad <= 200:
        parser.error('--cantidad debe estar entre 1 y 200')

    root = Path(__file__).resolve().parent
    out = args.salida if args.salida.is_absolute() else root / args.salida
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Generator().to(device)
    model.load_state_dict(torch.load(root / 'checkpoints/netG_hinge.pth',
                                    map_location=device, weights_only=True))
    # El checkpoint ya contiene las estadisticas BatchNorm usadas en la galeria.
    model.eval()
    print(f'Dispositivo: {device}; parametros del generador: '
          f'{sum(p.numel() for p in model.parameters()):,}')

    def generate(z):
        with torch.inference_mode():
            return (model(z.to(device)).cpu() * 0.5 + 0.5).clamp(0, 1)[0]

    gallery, labels, errors = [], [], {}
    for line in (root / 'galeria/semillas.txt').read_text().splitlines():
        name, _, seed_field = line.split()
        seed = int(seed_field.split('=')[1])
        z = torch.load(root / f'galeria/{name}_z.pt',
                       map_location='cpu', weights_only=True)
        generated = generate(z)
        image = to_image(generated)
        image.save(out / f'{name}_regenerado.png')
        with Image.open(root / f'galeria/{name}_seed_{seed}.png') as original:
            saved = torch.from_numpy(np.array(original.convert('RGB')).copy())
        saved = saved.permute(2, 0, 1).float() / 255
        error = float((generated - saved).abs().max())
        errors[name] = error
        print(f'{name}: error maximo = {error:.8f}')
        gallery.append(image)
        labels.append(f'{name} | seed {seed}')
    save_grid(gallery, labels, out / 'galeria_regenerada.png')

    images, labels = [], []
    for index in range(args.cantidad):
        seed = args.seed + index
        z = torch.randn(1, 100, 1, 1, generator=torch.Generator().manual_seed(seed))
        image = to_image(generate(z))
        image.save(out / f'nuevo_seed_{seed}.png')
        images.append(image)
        labels.append(f'Semilla {seed}')
    save_grid(images, labels, out / 'nuevos_personajes.png', columns=4)

    passed = all(error < 2 / 255 for error in errors.values())
    report = {'device': str(device), 'torch': torch.__version__,
              'errores_galeria': errors, 'galeria_reproducida': passed,
              'semillas_nuevas': list(range(args.seed, args.seed + args.cantidad))}
    (out / 'verificacion.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Galeria reproducida: {passed}')
    print(f'Resultados: {out}')
    if not passed:
        raise SystemExit('La galeria generada no coincide dentro de la tolerancia.')


if __name__ == '__main__':
    main()
