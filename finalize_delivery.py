"""Construye README, notebook ejecutable y un paquete coherente para Canvas."""
import argparse
from pathlib import Path
import json
import shutil
import zipfile
import nbformat
from gan_pipeline import ROOT, RUN, load_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=RUN)
    p.add_argument('--package',action='store_true')
    a=p.parse_args(); run=a.run.resolve(); relative=run.relative_to(ROOT).as_posix()
    selection=load_json(run/'galeria/manifest.json'); tag=selection['model']; epoch=selection['epoch']
    cfg=load_json(run/tag/'config.json'); lr_d=cfg.get('learning_rate_D',cfg['learning_rate'])
    indices=[r['candidate_index'] for r in selection['characters']]
    readme=f'''# Proyecto 2: Diseño generativo de personajes

**Luis Gonzalez — 23353**  
**Jonathan Zacarias — 231104**

GAN para conceptos de habitantes de criptas, cavernas y ruinas. La salida nativa es pixel art de 16×16, exportada a 64×64 con NEAREST. El informe de 12 páginas es `presentacion.pdf`; la versión editable es `presentacion_editable.docx`. La galería tiene diez conceptos, nombres, roles, semillas y vectores z. Son resultados experimentales: hay proporciones extrañas y repetición de arquetipos; no afirmamos que se haya demostrado ausencia de sobreajuste ni originalidad semántica completa.

## Ejecutar sin entrenar

Desde la carpeta del proyecto, con Python 3.14 (la ejecución se verificó en 3.14.6):

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python generar.py --verificar
python generar.py --seed 2026 --cantidad 16
python verify_project.py
```

La CPU basta para regenerar la galería. Las imágenes nuevas se guardan en `output_eval/nuevos/`. Cada semilla produce un z normal de tamaño 1×100×1×1; cambiar la semilla no entrena la red. `--verificar` comprueba pesos, diez PNG y z frente a sus semillas. Tolerancia de imagen: 2/255; el informe contiene el error real. La ejecución con nuevos números puede producir fallos visuales que la curación de diez ejemplos no muestra.

Para GPU NVIDIA compatible, instalamos los paquetes CUDA oficiales después de las demás dependencias:

```powershell
python -m pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

Hardware usado: {cfg['gpu']}, con 8 GB de VRAM. La CPU fue Intel Core i5-12400F. Las corridas completas y sus tiempos reales aparecen en los JSON. Diferentes dispositivos o versiones pueden introducir diferencias numéricas; las semillas no garantizan resultados idénticos entre todo hardware.

## Datos y fuentes

El paquete contiene los 312 PNG originales: 132 Tiny Dungeon (Kenney) y 180 Tiny Creatures (Clint Bellanger con colaboración de Kenney), ambas fuentes CC0. [Tiny Dungeon](https://kenney.nl/assets/tiny-dungeon), [Tiny Creatures](https://opengameart.org/content/tiny-creatures).

`prepare_data.py` revisa una lista explícita de personajes/criaturas, excluye objetos y una mano aislada, compone sobre negro, centra y escala con NEAREST. El manifiesto documenta las 312 decisiones y hashes originales. Conservamos 199 sprites: 164 de entrenamiento y 35 de validación. Variantes conocidas y siluetas iguales/reflejadas se agrupan antes de dividir (128 familias; 102/26). Solo entrenamiento recibe reflexión horizontal: 328 ejemplos. No usamos permutaciones RGB. Los 23 colores de la paleta se extraen exclusivamente de entrenamiento.

```powershell
python prepare_data.py
```

La preparación no requiere descargar el ZIP, porque `data_raw/` viene incluido. La agrupación manual puede omitir parentescos visuales. Validación se usa para seleccionar: no es un conjunto independiente de prueba.

## Entrenamiento y experimentos

`gan_pipeline.py` contiene Generator, Discriminator, alternancia Adam, BCE no saturante, Hinge y spectral_norm. El generador clasifica colores con argmax y utiliza un gradiente sustituto softmax durante entrenamiento. Las imágenes no provienen de un generador externo. La normalización espectral conserva BatchNorm para aislar su efecto.

La corrida seleccionada está en `{relative}/`: {cfg['epochs']} épocas por condición, con checkpoint seleccionado en la época {epoch}, lote 32, tasa G=0.0002, tasa D={lr_d}, betas (0.5,0.999). Comparamos BCE→Hinge cambiando la pérdida y Hinge→Hinge+SN cambiando solo spectral_norm. La inicialización, datos, ruido y condiciones restantes coinciden. `hypothesis.txt` se escribe antes del bucle de entrenamiento; `protocolo.md` registra las decisiones previas y exploraciones que no mejoraron.

Para repetir desde cero use una carpeta nueva, evitando sobrescribir resultados:

```powershell
python gan_pipeline.py train --tag base --epochs {cfg['epochs']} --run runs/reproduccion --palette --lr-d {lr_d}
python gan_pipeline.py train --tag hinge --epochs {cfg['epochs']} --run runs/reproduccion --palette --lr-d {lr_d}
python gan_pipeline.py train --tag hinge_sn --epochs {cfg['epochs']} --run runs/reproduccion --palette --lr-d {lr_d}
python gan_pipeline.py evaluate --tag base --epoch {epoch} --run runs/reproduccion --cpu
python gan_pipeline.py evaluate --tag hinge --epoch {epoch} --run runs/reproduccion --cpu
python gan_pipeline.py evaluate --tag hinge_sn --epoch {epoch} --run runs/reproduccion --cpu
python gan_pipeline.py export --tag {tag} --epoch {epoch} --run runs/reproduccion --indices {' '.join(map(str,indices))}
```

La evaluación CNN descarga una vez los pesos ImageNet oficiales de ResNet-18, si no están en caché. Usa entrada 224×224, características 512 y similitud coseno; RMS usa píxeles nativos. Se comparan contra los 164 originales y sus reflejos (328 referencias), y por separado contra 35 originales de validación. Estos indicadores tienen sesgos y no prueban que toda muestra sea nueva. Los checkpoints entregados contienen G y sus estadísticas de BatchNorm recalibradas, suficientes para inferencia; no incluyen D/estado Adam para reanudar entrenamiento. La reproducción del entrenamiento parte desde cero.

## Semillas y selección

Separación 1729; inicialización 42; orden de lotes 4200+época; ruido de entrenamiento 9000+época; ruido fijo 123 (16 vectores); sondas periódicas 777 (128); calibración BN 12345 (20 lotes de 128); candidatos 1000+índice (200 por condición). Cada personaje seleccionado conserva su z y semilla. El modelo final es **{tag}**, época **{epoch}**. Selección final 10/200 = **5%**. En todo el desarrollo se evaluaron {selection['development_candidate_count']} candidatos con distintos modelos/checkpoints: 10/{selection['development_candidate_count']} = **{100*selection['development_selection_rate']:.2f}%**. Se repiten semillas entre condiciones; además se inspeccionaron rejillas fijas. Esas cifras excluyen los lotes usados para aprender y las sondas periódicas.

La exportación canónica usa CPU, lote de inferencia 1. Al cambiar dispositivo o tamaño de lote, argmax puede elegir otro color en un píxel cercano a un empate. Una primera verificación desde GPU falló por ese motivo; su registro está conservado. Recalculamos candidatos y vecinos en CPU y la verificación final de las diez imágenes pasó. `generar.py` también usa CPU y lote 1.

`galeria/manifest.json` contiene nombres, roles, índices y SHA-256 del checkpoint. Una copia idéntica en `{relative}/galeria/` mantiene las rutas del pipeline. Los PNG evaluados son `personaje_01.png` a `personaje_10.png`; los mosaicos son únicamente figuras auxiliares.

## Archivos principales

- `proyecto-2.ipynb`: explicación, protocolo, ejecución opcional de experimentos, resultados reales y regeneración.
- `prepare_data.py`, `dataset/`: selección, familias, separación y paleta.
- `gan_pipeline.py`: redes, entrenamiento, evaluación y exportación.
- `generar.py`: inferencia y verificación de la galería.
- `verify_project.py`: integridad de datos y controles experimentales.
- `{relative}/`: configuraciones, historiales, hipótesis, pesos, evaluaciones y figuras.
- `protocolo.md`: decisiones previas y límites de los experimentos.
- `build_figures.py`, `build_report.py`: construcción del informe desde evidencia real.

Para reconstruir el PDF se requieren ReportLab y Arial instalado; en Windows:

```powershell
python -m pip install -r requirements-report.txt
python build_figures.py --run {relative} --epoch {epoch}
python build_report.py --run {relative} --epoch {epoch}
python build_docx.py --run {relative} --epoch {epoch}
```

Se incluyen historiales y rejillas de las etapas exploratorias, sin todos sus pesos intermedios para reducir el tamaño de entrega. El repositorio local conserva además los originales históricos. Los pesos finales de las tres condiciones y los checkpoints usados por la figura de evolución sí están en el ZIP.

## Versiones verificadas y asistentes de IA

Python 3.14.6; PyTorch 2.13.0+cu130; torchvision 0.28.0+cu130; NumPy 2.5.1; Pillow 11.3.0; Matplotlib 3.11.1; scikit-learn 1.9.0; SciPy 1.18.0; JupyterLab 4.6.1. El PDF se construyó con el runtime de documentos en Python 3.12.14; no interviene en el entrenamiento. Las versiones exactas se registran en `requirements.txt` y `environment.json`.

Usamos Codex y ChatGPT para comprender la GAN, revisar código y optimizadores, investigar fuentes, curar y agrupar datos, ejecutar experimentos y verificaciones, y redactar/maquetar el informe. El equipo debe comprender el código y defender los resultados. La IA no creó imágenes de la galería mediante un servicio externo.
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    (ROOT/'requirements-report.txt').write_text('reportlab>=4.4,<5\npypdf>=6,<7\npython-docx>=1.2,<2\n',encoding='utf-8')
    nb=nbformat.v4.new_notebook(); cells=[]
    def md(text): cells.append(nbformat.v4.new_markdown_cell(text))
    def code(text): cells.append(nbformat.v4.new_code_cell(text))
    md('# Proyecto 2: Diseño generativo de personajes\n\nLuis Gonzalez - 23353  \nJonathan Zacarias - 231104\n\nUniverso: habitantes de criptas, cavernas y ruinas en pixel art. Este notebook usa módulos legibles del repositorio. Por defecto inspecciona resultados y regenera la galería; entrenar y evaluar son pasos opcionales. El informe reconoce repetición de arquetipos, inestabilidad y límites para demostrar novedad.')
    code('from pathlib import Path\nimport json, sys, subprocess\nimport torch\nfrom IPython.display import display, Image\nfrom gan_pipeline import ROOT, RUN, load_json, load_generator, sample\nprint("Python:", sys.version.split()[0], "PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())\nprint("Corrida:", RUN)')
    md('## Datos, licencia y separación\n\nTiny Dungeon de Kenney y Tiny Creatures de Clint Bellanger/Kenney: CC0. 312 originales; 199 aceptados; 164 train y 35 validación. La revisión elimina escenarios/objetos y una mano aislada. Las familias conocidas y siluetas idénticas/reflejadas permanecen juntas. Solo entrenamiento recibe reflexión; no se permutan colores. `prepare_data.py` reconstruye las decisiones y `dataset/manifest.csv` las registra por archivo.')
    code('summary=load_json(ROOT/"dataset/summary.json")\nprint(json.dumps(summary,indent=2,ensure_ascii=False))\ndisplay(Image(filename=str(ROOT/"dataset/train_grid.png")))\ndisplay(Image(filename=str(ROOT/"dataset/validation_grid.png")))')
    md('## Cómo aprende y cómo genera\n\nDurante entrenamiento, D aprende a separar reales y falsas; G recibe a través de D una señal para cambiar sus pesos y hacer más plausibles sus salidas. Cada red tiene su propio Adam. La rejilla aprendida es 16×16, con 23 colores y gradiente sustituto softmax para el argmax. Al generar, G recibe ruido nuevo con los pesos congelados; D no participa y no hay aprendizaje. El diagrama representa ambas etapas.')
    code('display(Image(filename=str(RUN/"figures/flujo.png")))\nmanifest=load_json(RUN/"galeria/manifest.json")\nmodel,payload=load_generator(ROOT/manifest["checkpoint"],torch.device("cpu"))\nprint(model)\nprint("Parámetros G:",sum(p.numel() for p in model.parameters()))')
    md('## Hipótesis previas y controles\n\nCreemos que Hinge mejorará porque usa un margen en D y una pérdida lineal en G. Lo consideraremos útil si preserva diversidad y produce siluetas más claras con el mismo ruido fijo, sin aumentar similitud excesiva con entrenamiento.\n\nCreemos que la normalización espectral mejorará porque limita la amplificación de las convoluciones de D. Lo consideraremos útil si mejora las muestras y evita dominancia prolongada de D. Conservamos BatchNorm.\n\nEstas hipótesis están en `hypothesis.txt`, escrito antes de cada bucle ejecutado, y en `protocolo.md`. Esta versión organizada del notebook reúne esa evidencia. BCE/Hinge cambia solo pérdida; Hinge/SN cambia solo normalización espectral. Las etapas RGB, de paleta y de tasa de D se explican como exploraciones separadas.')
    code('for tag in ["base","hinge","hinge_sn"]:\n    cfg=load_json(RUN/tag/"config.json")\n    print(tag, "loss=",cfg["loss"],"SN=",cfg["spectral_norm"],"épocas=",cfg["epochs"],"tasa D=",cfg.get("learning_rate_D",cfg["learning_rate"]))\n    print((RUN/tag/"hypothesis.txt").read_text(encoding="utf-8"))')
    md('## Repetir entrenamiento desde cero\n\nLa ejecución está desactivada por defecto. Cambie `REENTRENAR` a True solo para crear tres corridas nuevas. Los pesos entregados sirven para generar; no reanudan Adam ni D. Las condiciones siguientes conservan todos los controles. La carpeta nueva evita sobrescribir evidencia.')
    code(f'REENTRENAR=False\nNUEVA_CORRIDA=ROOT/"runs/reproduccion"\nif REENTRENAR:\n    for tag in ["base","hinge","hinge_sn"]:\n        subprocess.run([sys.executable,"gan_pipeline.py","train","--tag",tag,"--epochs","{cfg["epochs"]}","--run",str(NUEVA_CORRIDA),"--palette","--lr-d","{lr_d}"],cwd=ROOT,check=True)\nelse:\n    print("Se usan las corridas guardadas; no se vuelve a entrenar.")')
    md('## Evolución y pérdidas\n\nLos 16 vectores fijos de semilla 123 son iguales entre condiciones y épocas. Una D cerca de cero durante muchas épocas indica separación fácil, no éxito. La diversidad en píxeles puede aumentar por ruido. Las escalas BCE/Hinge no son directamente comparables.')
    code('display(Image(filename=str(RUN/"figures/fixed_evolution_selected.png")))\ndisplay(Image(filename=str(RUN/"figures/all_losses.png")))\ndecision=load_json(RUN/"selection.json")\nprint(decision["training_conclusion"])\nprint(decision["stabilization_conclusion"])')
    md('## Evaluar y elegir\n\nEvaluamos 200 semillas por condición (1000–1199) con RMS nativo y ResNet-18 preentrenada a 224×224. Los vecinos consideran 328 referencias train incluidas sus reflexiones. Validación no recibe gradientes pero se usa para elegir; no constituye prueba independiente. ResNet se entrenó en fotografías. Una sola semilla experimental y ninguna métrica prueban ausencia de memorización.')
    code(f'for tag in ["base","hinge","hinge_sn"]:\n    metrics=load_json(RUN/tag/"evaluation_epoch_{epoch:04d}.json")\n    print(tag,{{key:metrics[key] for key in ["diversity","nearest_train_rms_mean","nearest_validation_rms_mean","resnet_cosine_mean"]}})\nprint(decision["model_choice"])\ndisplay(Image(filename=str(RUN/"figures/model_comparison.png")))')
    code(f'REEVALUAR=False\nif REEVALUAR:\n    for tag in ["base","hinge","hinge_sn"]:\n        subprocess.run([sys.executable,"gan_pipeline.py","evaluate","--tag",tag,"--epoch","{epoch}","--run",str(RUN),"--cpu"],cwd=ROOT,check=True)')
    md(f'## Galería y vecinos\n\nDiez de 200 candidatos de la condición elegida: 5%. En el desarrollo se evaluaron {selection["development_candidate_count"]} candidatos con distintos modelos y checkpoints, repitiendo semillas: 10/{selection["development_candidate_count"]} = {100*selection["development_selection_rate"]:.2f}%. Además se inspeccionaron rejillas fijas. Los nombres y roles son interpretaciones humanas; cada PNG sale del modelo. No consideramos diez semillas distintas como prueba automática de diez personajes semánticamente distintos. La exportación canónica usa CPU y lote 1 para evitar cambios de color del argmax entre dispositivos.')
    code('display(Image(filename=str(RUN/"galeria/galeria_mosaico.png")))\ndisplay(Image(filename=str(RUN/"galeria/vecinos_rms.png")))\ndisplay(Image(filename=str(RUN/"galeria/vecinos_resnet.png")))\nprint(decision["gallery_conclusion"])\nprint(decision["reflection"])')
    md('## Regenerar y comprobar\n\nLa verificación usa CPU y compara cada PNG y vector guardado con su semilla. La auditoría también comprueba hashes originales, división por grupos, duplicados y controles experimentales.')
    code('subprocess.run([sys.executable,"generar.py","--verificar"],cwd=ROOT,check=True)\nsubprocess.run([sys.executable,"verify_project.py"],cwd=ROOT,check=True)')
    md('## Generar imágenes nuevas\n\nEste paso crea una demostración con semillas nuevas, sin entrenar. Los resultados no se filtran; pueden mostrar fallos no presentes en la galería curada.')
    code('subprocess.run([sys.executable,"generar.py","--seed","2026","--cantidad","16"],cwd=ROOT,check=True)\ndisplay(Image(filename=str(ROOT/"output_eval/nuevos/mosaico.png")))')
    md('## Referencias y uso de IA\n\n[DCGAN](https://arxiv.org/abs/1511.06434), [GAN](https://arxiv.org/abs/1406.2661), [normalización espectral](https://arxiv.org/abs/1802.05957), [gradientes sustitutos](https://arxiv.org/abs/1308.3432). Codex y ChatGPT ayudaron a revisar código, investigar, depurar datos, ejecutar pruebas y redactar el reporte. Ninguna imagen evaluada proviene de un generador externo. El equipo debe comprender y defender el proceso y sus limitaciones.')
    nb.cells=cells;nb.metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python','version':'3.14.6'}}
    if not a.package: nbformat.write(nb,ROOT/'proyecto-2.ipynb')
    if not a.package: print('README y notebook actualizados');return
    shutil.copy2(ROOT/'output/pdf/presentacion.pdf',ROOT/'presentacion.pdf')
    # Mantener copia de la galeria historica; destino comprobado dentro de este repo.
    historical=ROOT/'historico/galeria_original'; historical.mkdir(parents=True,exist_ok=True)
    galleryroot=ROOT/'galeria'; galleryroot.mkdir(exist_ok=True)
    for old in galleryroot.iterdir():
        if old.is_file() and not (historical/old.name).exists(): shutil.copy2(old,historical/old.name)
    for old in galleryroot.iterdir():
        if old.is_file(): old.unlink()
    for source in (run/'galeria').iterdir():
        if source.is_file(): shutil.copy2(source,galleryroot/source.name)
    stage=ROOT/'entrega_final'; stage.mkdir(exist_ok=True)
    names=['README.md','requirements.txt','requirements-report.txt','proyecto-2.ipynb','presentacion.pdf','presentacion_editable.docx','prepare_data.py','gan_pipeline.py','generar.py','verify_project.py','build_figures.py','build_report.py','build_docx.py','protocolo.md','finalize_delivery.py','environment.json']
    for name in names:
        source=ROOT/name
        if source.exists(): shutil.copy2(source,stage/name)
    for name in ['dataset','data_raw','galeria']:
        shutil.copytree(ROOT/name,stage/name,dirs_exist_ok=True)
    # Los pesos finales y los usados en la figura de evolucion; otras etapas solo evidencia.
    for source_run in (ROOT/'runs').iterdir():
        if not source_run.is_dir():continue
        is_final=source_run==run
        for source in source_run.rglob('*'):
            if not source.is_file():continue
            keep=source.suffix.lower() in {'.json','.txt','.png','.csv','.md'}
            if source.suffix=='.pt':
                keep=is_final and (source.name in {'fixed_noise.pt',f'generator_epoch_{epoch:04d}.pt',f'candidates_epoch_{epoch:04d}.pt',f'neighbors_epoch_{epoch:04d}.pt'} or
                    (source.parent.name==tag and source.name in ['generator_epoch_0001.pt','generator_epoch_0100.pt','generator_epoch_0500.pt']) or source.parent.name=='galeria')
            if keep:
                destination=stage/source.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    if (ROOT/'output_eval/nuevos').exists():shutil.copytree(ROOT/'output_eval/nuevos',stage/'output_eval/nuevos',dirs_exist_ok=True)
    manifest=[]
    import hashlib
    for source in sorted(stage.rglob('*')):
        if source.is_file() and source.name!='SHA256SUMS.txt':manifest.append(f'{hashlib.sha256(source.read_bytes()).hexdigest()}  {source.relative_to(stage).as_posix()}')
    (stage/'SHA256SUMS.txt').write_text('\n'.join(manifest)+'\n',encoding='utf-8')
    zipname=ROOT/'entrega_final.zip'
    with zipfile.ZipFile(zipname,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for source in sorted(stage.rglob('*')):
            if source.is_file():z.write(source,source.relative_to(stage))
    print('Paquete:',zipname,'MB:',round(zipname.stat().st_size/1024**2,2))


if __name__=='__main__':main()
