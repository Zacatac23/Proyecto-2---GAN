# Proyecto 2: Diseño generativo de personajes

**Luis Gonzalez — 23353**  
**Jonathan Zacarias — 231104**

GAN para conceptos de habitantes de criptas, cavernas y ruinas. La salida nativa es pixel art de 16×16, exportada a 64×64 con NEAREST. El informe de 12 páginas es `presentacion.pdf`; la versión editable es `presentacion_editable.docx`. La galería tiene diez conceptos, nombres, roles, semillas y vectores z. Son resultados experimentales: hay proporciones extrañas y repetición de arquetipos; no afirmamos que se haya demostrado ausencia de sobreajuste ni originalidad semántica completa.

## Ejecutar sin entrenar

Desde la carpeta del proyecto, con Python 3.14 (la ejecución se verificó en 3.14.6):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
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

Hardware usado: NVIDIA GeForce RTX 5050, con 8 GB de VRAM. La CPU fue Intel Core i5-12400F. Las corridas completas y sus tiempos reales aparecen en los JSON. Diferentes dispositivos o versiones pueden introducir diferencias numéricas; las semillas no garantizan resultados idénticos entre todo hardware.

## Datos y fuentes

El paquete contiene los 312 PNG originales: 132 Tiny Dungeon (Kenney) y 180 Tiny Creatures (Clint Bellanger con colaboración de Kenney), ambas fuentes CC0. [Tiny Dungeon](https://kenney.nl/assets/tiny-dungeon), [Tiny Creatures](https://opengameart.org/content/tiny-creatures).

`prepare_data.py` revisa una lista explícita de personajes/criaturas, excluye objetos y una mano aislada, compone sobre negro, centra y escala con NEAREST. El manifiesto documenta las 312 decisiones y hashes originales. Conservamos 199 sprites: 164 de entrenamiento y 35 de validación. Variantes conocidas y siluetas iguales/reflejadas se agrupan antes de dividir (128 familias; 102/26). Solo entrenamiento recibe reflexión horizontal: 328 ejemplos. No usamos permutaciones RGB. Los 23 colores de la paleta se extraen exclusivamente de entrenamiento.

```powershell
python prepare_data.py
```

La preparación no requiere descargar el ZIP, porque `data_raw/` viene incluido. La agrupación manual puede omitir parentescos visuales. Validación se usa para seleccionar: no es un conjunto independiente de prueba.

## Entrenamiento y experimentos

`gan_pipeline.py` contiene Generator, Discriminator, alternancia Adam, BCE no saturante, Hinge y spectral_norm. El generador clasifica colores con argmax y utiliza un gradiente sustituto softmax durante entrenamiento. Las imágenes no provienen de un generador externo. La normalización espectral conserva BatchNorm para aislar su efecto.

La corrida seleccionada está en `runs/paleta_2000/`: 2000 épocas por condición, con checkpoint seleccionado en la época 1500, lote 32, tasa G=0.0002, tasa D=0.0002, betas (0.5,0.999). Comparamos BCE→Hinge cambiando la pérdida y Hinge→Hinge+SN cambiando solo spectral_norm. La inicialización, datos, ruido y condiciones restantes coinciden. `hypothesis.txt` se escribe antes del bucle de entrenamiento; `protocolo.md` registra las decisiones previas y exploraciones que no mejoraron.

Para repetir desde cero use una carpeta nueva, evitando sobrescribir resultados:

```powershell
python gan_pipeline.py train --tag base --epochs 2000 --run runs/reproduccion --palette --lr-d 0.0002
python gan_pipeline.py train --tag hinge --epochs 2000 --run runs/reproduccion --palette --lr-d 0.0002
python gan_pipeline.py train --tag hinge_sn --epochs 2000 --run runs/reproduccion --palette --lr-d 0.0002
python gan_pipeline.py evaluate --tag base --epoch 1500 --run runs/reproduccion --cpu
python gan_pipeline.py evaluate --tag hinge --epoch 1500 --run runs/reproduccion --cpu
python gan_pipeline.py evaluate --tag hinge_sn --epoch 1500 --run runs/reproduccion --cpu
python gan_pipeline.py export --tag base --epoch 1500 --run runs/reproduccion --indices 0 8 20 26 35 70 72 17 114 171
```

La evaluación CNN descarga una vez los pesos ImageNet oficiales de ResNet-18, si no están en caché. Usa entrada 224×224, características 512 y similitud coseno; RMS usa píxeles nativos. Se comparan contra los 164 originales y sus reflejos (328 referencias), y por separado contra 35 originales de validación. Estos indicadores tienen sesgos y no prueban que toda muestra sea nueva. Los checkpoints entregados contienen G y sus estadísticas de BatchNorm recalibradas, suficientes para inferencia; no incluyen D/estado Adam para reanudar entrenamiento. La reproducción del entrenamiento parte desde cero.

## Semillas y selección

Separación 1729; inicialización 42; orden de lotes 4200+época; ruido de entrenamiento 9000+época; ruido fijo 123 (16 vectores); sondas periódicas 777 (128); calibración BN 12345 (20 lotes de 128); candidatos 1000+índice (200 por condición). Cada personaje seleccionado conserva su z y semilla. El modelo final es **base**, época **1500**. Selección final 10/200 = **5%**. En todo el desarrollo se evaluaron 2400 candidatos con distintos modelos/checkpoints: 10/2400 = **0.42%**. Se repiten semillas entre condiciones; además se inspeccionaron rejillas fijas. Esas cifras excluyen los lotes usados para aprender y las sondas periódicas.

La exportación canónica usa CPU, lote de inferencia 1. Al cambiar dispositivo o tamaño de lote, argmax puede elegir otro color en un píxel cercano a un empate. Una primera verificación desde GPU falló por ese motivo; su registro está conservado. Recalculamos candidatos y vecinos en CPU y la verificación final de las diez imágenes pasó. `generar.py` también usa CPU y lote 1.

`galeria/manifest.json` contiene nombres, roles, índices y SHA-256 del checkpoint. Una copia idéntica en `runs/paleta_2000/galeria/` mantiene las rutas del pipeline. Los PNG evaluados son `personaje_01.png` a `personaje_10.png`; los mosaicos son únicamente figuras auxiliares.

## Archivos principales

- `proyecto-2.ipynb`: explicación, protocolo, ejecución opcional de experimentos, resultados reales y regeneración.
- `prepare_data.py`, `dataset/`: selección, familias, separación y paleta.
- `gan_pipeline.py`: redes, entrenamiento, evaluación y exportación.
- `generar.py`: inferencia y verificación de la galería.
- `verify_project.py`: integridad de datos y controles experimentales.
- `runs/paleta_2000/`: configuraciones, historiales, hipótesis, pesos, evaluaciones y figuras.
- `protocolo.md`: decisiones previas y límites de los experimentos.
- `build_figures.py`, `build_report.py`: construcción del informe desde evidencia real.

Para reconstruir el PDF se requieren ReportLab y Arial instalado; en Windows:

```powershell
python -m pip install -r requirements-report.txt
python build_figures.py --run runs/paleta_2000 --epoch 1500
python build_report.py --run runs/paleta_2000 --epoch 1500
python build_docx.py --run runs/paleta_2000 --epoch 1500
```

Se incluyen historiales y rejillas de las etapas exploratorias, sin todos sus pesos intermedios para reducir el tamaño de entrega. El repositorio local conserva además los originales históricos. Los pesos finales de las tres condiciones y los checkpoints usados por la figura de evolución sí están en el ZIP.

## Versiones verificadas y asistentes de IA

Python 3.14.6; PyTorch 2.13.0+cu130; torchvision 0.28.0+cu130; NumPy 2.5.1; Pillow 11.3.0; Matplotlib 3.11.1; scikit-learn 1.9.0; SciPy 1.18.0; JupyterLab 4.6.1. El PDF se construyó con el runtime de documentos en Python 3.12.14; no interviene en el entrenamiento. Las versiones exactas se registran en `requirements.txt` y `environment.json`.

Usamos Codex y ChatGPT para comprender la GAN, revisar código y optimizadores, investigar fuentes, curar y agrupar datos, ejecutar experimentos y verificaciones, y redactar/maquetar el informe. El equipo debe comprender el código y defender los resultados. La IA no creó imágenes de la galería mediante un servicio externo.
