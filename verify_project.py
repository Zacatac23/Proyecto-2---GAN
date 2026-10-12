"""Comprueba datos, controles experimentales y regenerabilidad de la entrega."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from gan_pipeline import ROOT, RUN, VARIANTS, verify_gallery, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=RUN)
    args=parser.parse_args(); run=args.run.resolve()
    rows=json.loads((ROOT/'dataset/manifest.json').read_text(encoding='utf-8'))
    accepted=[r for r in rows if r['keep']]
    assert len({r['raw_path'] for r in rows})==len(rows)
    train={r['group'] for r in accepted if r['split']=='train'}
    validation={r['group'] for r in accepted if r['split']=='validation'}
    assert not train&validation
    fingerprints=set()
    for row in rows:
        assert hashlib.sha256((ROOT/row['raw_path']).read_bytes()).hexdigest()==row['raw_sha256']
        if not row['keep']: continue
        pixels=np.array(Image.open(ROOT/row['image_path']).convert('RGB'))
        assert pixels.shape==(64,64,3)
        assert np.array_equal(pixels,np.repeat(np.repeat(pixels[::4,::4],4,0),4,1))
        fingerprint=min(hashlib.sha256(pixels.tobytes()).hexdigest(),hashlib.sha256(pixels[:,::-1].tobytes()).hexdigest())
        assert fingerprint not in fingerprints
        fingerprints.add(fingerprint)
    configs={tag:json.loads((run/tag/'config.json').read_text(encoding='utf-8')) for tag in VARIANTS}
    shared=['epochs','seed','latent_dim','width','batch_size','learning_rate','betas','native_resolution',
            'export_resolution','upsampling','discriminator_batchnorm','initial_G_sha256','initial_D_sha256',
            'batch_order_seed','training_noise_seed','fixed_noise_seed','evaluation_noise_seed',
            'bn_calibration_seed','bn_calibration_batches','train_images','augmented_images',
            'validation_images','manifest_sha256','palette','palette_training']
    for key in shared:
        assert all(config[key]==configs['base'][key] for config in configs.values()),key
    for key,default in [('learning_rate_D',2e-4),('discriminator_width',64)]:
        assert all(config.get(key,default)==configs['base'].get(key,default) for config in configs.values()),key
    assert configs['base']['loss']=='bce' and not configs['base']['spectral_norm']
    assert configs['hinge']['loss']=='hinge' and not configs['hinge']['spectral_norm']
    assert configs['hinge_sn']['loss']=='hinge' and configs['hinge_sn']['spectral_norm']
    for tag,config in configs.items():
        history=json.loads((run/tag/'history.json').read_text(encoding='utf-8'))
        assert [r['epoch'] for r in history]==list(range(1,config['epochs']+1))
        assert all(np.isfinite([r['D'],r['G'],r['real_score'],r['fake_score']]).all() for r in history)
        assert (run/tag/'hypothesis.txt').stat().st_size>0
    verify_gallery(run)
    result={'passed':True,'raw_files_verified':len(rows),'unique_selected_images':len(accepted),
            'train_validation_groups_disjoint':True,'experimental_controls_identical':shared,
            'epochs_per_condition':configs['base']['epochs'],'gallery_cpu_regeneration':True}
    write_json(run/'audit.json',result)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__': main()
