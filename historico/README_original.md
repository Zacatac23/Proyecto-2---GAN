# Proyecto 2: Diseño Generativo de Personajes (GAN Pixel Art)

**Curso:** Deep Learning (2026) — Universidad del Valle  
**Catedrático:** Kevin Recinos  
**Autores:** Jonathan Zacarias, Gilberto González  
**Fecha de entrega:** 11 de octubre de 2026  

---

## 1. Definición del Universo Visual
El proyecto explora la generación procedural de entidades para un juego de mazmorras (*dungeon crawler*) en estilo **pixel art clásico**. 

* **Estilo y estética:** Sprites de baja resolución ($16\times16$ reescalados a $64\times64$ con interpolación `NEAREST` para conservar bordes definidos de píxel), paleta reducida, contornos exteriores oscuros y fondo plano negro/oscuro.
* **Ambiente:** Mazmorras subterráneas, criptas y ruinas fantásticas habitadas por aventureros, criaturas místicas, no-muertos y aberraciones arcanas.
* **Rasgos esperados:** Siluetas centradas, masas redondeadas o antropomorfas, contraste marcado contra el fondo y proporciones compactas.

---

## 2. Galería de Personajes (Bonificación de Lore: Nombres y Roles)
Se generaron **200 candidatos** a partir de vectores de ruido latente generados con semillas deterministas (`1000 + índice`). De estos, se seleccionaron **10 personajes (Tasa de selección: 5% / 10 de 200)** con base en claridad de silueta, postura central, variedad cromática y contraste.

A continuación se detalla la ficha de cada personaje para la bonificación de universo y rol:

| # | Archivo | Nombre | Rol / Clase | Índice | Semilla / Vector $z$ | Descripción y Rasgos en el Universo |
|---|---|---|---|---|---|---|
| **01** | `personaje_01.png` | **Morvath el Errante** | Conjurador Sombrío | 4 | `personaje_01_z.pt` (seed 1004) | Mago ermitaño con túnica oscura rasgada y resplandor espectral en el torso. Especialista en maleficios arcanos. |
| **02** | `personaje_02.png` | **Ignis el Ígneo** | Elemental de Magma | 14 | `personaje_02_z.pt` (seed 1014) | Núcleo de roca volcánica flotante con tonos cálidos y amarillentos. Custodio de las forjas olvidadas. |
| **03** | `personaje_03.png` | **Grog Rompehuesos** | Guerrero Ogro | 29 | `personaje_03_z.pt` (seed 1029) | Berserker acorazado de complexión ancha y silueta pesada. Vanguardia de asalto en pasadizos estrechos. |
| **04** | `personaje_04.png` | **Kaelen de la Guardia** | Paladín de Cripta | 45 | `personaje_04_z.pt` (seed 1045) | Campeón cubierto con peto plateado y capa protectora. Guardián de reliquias sagradas subterráneas. |
| **05** | `personaje_05.png` | **Sombra Menor** | Espectro Rastrero | 64 | `personaje_05_z.pt` (seed 1064) | Entidad no-muerta etérea que se desliza pegada al suelo. Destaca por su alta proporción de tonos oscuros y halo difuso. |
| **06** | `personaje_06.png` | **Zul'Karr el Fétido** | Brujo Goblin | 104 | `personaje_06_z.pt` (seed 1104) | Chamán de baja estatura con pigmentación verdosa y cayado ritual. Envenena trampas y pasadizos. |
| **07** | `personaje_07.png` | **Glacius el Frío** | Espíritu de Escarcha | 110 | `personaje_07_z.pt` (seed 1110) | Manifestación elemental gélida de tonalidad celeste pálida. Ralentiza a invasores en las catacumbas congeladas. |
| **08** | `personaje_08.png` | **Mossgolem** | Bestia del Pantano | 112 | `personaje_08_z.pt` (seed 1112) | Criatura cuadrúpeda / humanoide compacta cubierta de musgo denso. Silueta inspirada en la fauna salvaje de las cloacas. |
| **09** | `personaje_09.png` | **Aveline del Viento** | Exploradora Pícara | 154 | `personaje_09_z.pt` (seed 1154) | Cazadora ágil con vestidura de cuero ligero y postura ladeada. Detección de trampas y ataques furtivos. |
| **10** | `personaje_10.png` | **Ursok el Terráneo** | Guardián Bestial | 184 | `personaje_10_z.pt` (seed 1184) | Fiera de pelaje pardo y orejas puntiagudas prominentes. Defensor territorial de las cavernas inferiores. |

---

## 3. Datos y Licencias
* **Fuentes originales:**
  * *Tiny Dungeon* por Kenney ([opengameart.org/content/tiny-dungeon](https://opengameart.org/content/tiny-dungeon)) — **Licencia: CC0 (Dominio Público)**.
  * *Tiny Creatures* por Clint Bellanger ([opengameart.org/content/tiny-creatures](https://opengameart.org/content/tiny-creatures)) — **Licencia: CC0 (Dominio Público)**.
* **Preprocesamiento:**
  * Extracción de 312 sprites $\rightarrow$ Depuración manual a **200 sprites limpios** de $16\times16$.
  * Padding simétrico a relación 1:1, reescalado a $64\times64$ con interpolación `NEAREST` para evitar artefactos de interpolación bilineal/bicúbica.
  * Aumento de datos ($\times 12$): Inversión horizontal ($2\times$) $\times$ 6 permutaciones de canales de color RGB $\rightarrow$ **2,400 imágenes de entrenamiento**.

---

## 4. Arquitectura y Experimentos

### 4.1. Configuración Base (DCGAN)
* **Generador:** Entrada $z \in \mathbb{R}^{100} \rightarrow$ Convoluciones transpuestas (`ConvTranspose2d`) con `BatchNorm2d` y `ReLU` $\rightarrow$ Salida $64\times64\times3$ con `Tanh`.
* **Discriminador:** `Conv2d` con `LeakyReLU(0.2)` y `BatchNorm2d`, sin Sigmoid final (salida en logits).
* **Parámetros:** Optimizador Adam ($\text{lr} = 0.0002$, $\beta_1 = 0.5$, $\beta_2 = 0.999$), Batch Size = 64, Épocas = 60.

### 4.2. Experimento de Función de Pérdida (BCE vs. Hinge)
* **Hipótesis:** Hinge loss evitará gradientes saturados al no penalizar al discriminador una vez supera el margen establecido, otorgando al generador un gradiente más suave y previniendo colapsos.
* **Resultado:** Hinge estabilizó la varianza de $G$, pero $D$ saturó con pérdidas cercanas a cero entre las épocas 30 y 55. Diversidad media: 0.333 vs 0.331 (BCE). Modelo final adoptado: **Hinge** (`netG_hinge.pth`).

### 4.3. Experimento de Estabilización (Spectral Normalization)
* **Hipótesis:** Limitar la constante de Lipschitz del discriminador mediante normalización espectral prevendría su dominancia temprana sin depender de las estadísticas de lote.
* **Resultado / Limitación metodológica:** Las curvas de pérdida mostraron un equilibrio teórico perfecto, pero las muestras generadas resultaron oscuras y de menor diversidad (0.268). Al implementar `spectral_norm` se retiró `BatchNorm2d`, modificando dos variables simultáneamente.

---

## 5. Instrucciones de Reproducción y Regenerabilidad

### 5.1. Requisitos de Entorno
* Python 3.10+
* PyTorch $\ge$ 2.0.0
* Torchvision $\ge$ 0.15.0
* Pillow, NumPy, Matplotlib, SciPy

Instalación de dependencias:
```bash
pip install torch torchvision numpy pillow matplotlib scipy
```

### 5.2. Semillas Utilizadas
* Semilla global de entrenamiento: `42`
* Semilla de ruido fijo para monitoreo (16 vectores): `123` (`checkpoints/fixed_noise.pt`)
* Semillas de candidatos de galería: `1000 + i` para $i \in [0, 199]$

### 5.3. Regenerar la Galería Exacta
Para regenerar los 10 personajes idénticos a los entregados en `galeria/`, ejecute la celda final del notebook `proyecto-2.ipynb` o el script de verificación. El generador cargará los pesos de `checkpoints/netG_hinge.pth` y aplicará `recalibrate_bn`:

```python
import torch
from torchvision.utils import save_image

# Cargar generador y recalibrar estadísticas de BatchNorm
netG = Generator().to(device)
netG.load_state_dict(torch.load("checkpoints/netG_hinge.pth", map_location=device))
recalibrate_bn(netG, device=device)
netG.eval()

# Cargar vector latente y generar
z = torch.load("galeria/personaje_01_z.pt", map_location=device)
with torch.no_grad():
    img_regen = netG(z)
save_image(img_regen * 0.5 + 0.5, "galeria/personaje_01_regenerado.png")
```
*Diferencia absoluta máxima verificada frente a la entrega: $\le 0.00196$ (asociada exclusivamente al redondeo entero de 8 bits en PNG).*

---

## 6. Enlace a Pesos del Modelo
Los pesos del generador final están almacenados localmente en el repositorio:
* Ruta: `checkpoints/netG_hinge.pth` (Tamaño aproximado: ~14 MB, apto para entrega directa en Canvas).
* *Enlace de respaldo en la nube:* `[Insertar enlace de Google Drive / OneDrive si Canvas restringe tamaño]`

---

## 7. Declaración de Asistentes de IA
En cumplimiento con la **Sección 8** del enunciado:
* Se utilizaron modelos de asistencia de IA (LLMs) como apoyo para la estructuración de la rutina de interpolación/recalibración de `BatchNorm`, refinamiento de funciones de cálculo de métricas de distancia (RMS y similitud coseno con ResNet-18) y redacción técnica de la memoria del proyecto.
* **Verificación de autenticidad:** Todo el código fue ejecutado y validado en nuestro entorno local. **Ninguna imagen de la galería ni de los experimentos proviene de herramientas de generación externa (Midjourney, DALL-E, Stable Diffusion, etc.)**; todas fueron sintetizadas de manera estricta por el modelo DCGAN entrenado por el equipo.