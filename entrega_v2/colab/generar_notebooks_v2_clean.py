#!/usr/bin/env python
"""Genera los DOS notebooks de Colab del dataset LIMPIO (v2 desde cero).

Mismo patron que ``generar_notebooks_v2.py``: las celdas se escriben desde aqui
para poder revisarlas y corregirlas en un solo sitio, sin editar JSON a mano.

    python colab/generar_notebooks_v2_clean.py

Escribe, junto a este archivo:

    ajedrez_v2_detector_desde_cero.ipynb
    ajedrez_v2_clasificador_desde_cero.ipynb

Diferencias con los notebooks anteriores (los del lote en cuarentena):

* DETECTOR: ``MODO = "desde_cero"`` y base ``yolo11n.pt`` (COCO). Antes se
  afinaba desde ``modelo_sintetico.pt``, que se entreno con la convencion vieja
  de fichas y con los compuestos cruzados. Ya no se usa como base: un peso
  contaminado contamina el reentreno.
* DETECTOR: 120 epocas, batch 16, lr0 0.01 (desde cero necesita mas learning
  rate que el fine-tune, que usaba 0.001).
* CLASIFICADOR: ``weights=IMAGENET`` SIN congelar y AMP (precision mixta),
  20 epocas con pesos de clase por frecuencia inversa.
* Todas las celdas de codigo son Python puro: no hay magics (``!`` ni ``%``),
  asi el notebook se puede auditar con ``ast.parse`` celda por celda.
"""
from __future__ import annotations

import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent


def cuerpo(texto: str) -> list[str]:
    """nbformat espera una lista de lineas; cada una conserva su \\n."""
    lineas = texto.strip("\n").split("\n")
    return [linea + "\n" for linea in lineas[:-1]] + [lineas[-1]]


def md(texto: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": cuerpo(texto)}


def code(texto: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cuerpo(texto),
    }


def metadatos(nombre: str) -> dict:
    return {
        "accelerator": "GPU",
        "colab": {"name": nombre, "provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }


# ===========================================================================
# DETECTOR
# ===========================================================================
def notebook_detector() -> dict:
    celdas = [
        md(
            """# Detector de piezas v2 — DESDE CERO (dataset limpio)

Este notebook entrena **(a) el detector YOLO** de la arquitectura v2 con el
dataset reconstruido desde la posición inicial física correcta.

| | |
|---|---|
| Modelo base | `yolo11n.pt` (COCO, `MODO = "desde_cero"`) |
| Paquete | `detector_v2_clean.zip` (~287 MB) en `Mi unidad/ajedrez_v2/` |
| Train | 4,044 tableros generados: 1,544 sintéticos + 2,500 compuestos limpios |
| Valid | 109 recortes **reales** (01/02/03/05, sin repetir posición) |
| Test | 110 recortes **reales** (02/03/04/05 + parte de 01) |
| Épocas | 120, `batch=16`, `imgsz=640`, `lr0=0.01` |

**Por qué desde cero y no desde `modelo_sintetico.pt`:** ese peso se entrenó con
la convención vieja de fichas y los compuestos heredaron el cruce caballo/alfil.
Un peso contaminado contamina el reentreno: aquí se parte de COCO.

**La regla que no se toca:** `valid` y `test` son SOLO fotos reales de la cámara.
El train es todo lo generado. Si val o test tuvieran imágenes sintéticas, la
métrica mediría lo que el modelo ya vio y no serviría para decidir nada.

Antes de ejecutar: **Entorno de ejecución > Cambiar tipo de entorno > T4 GPU**."""
        ),
        md("## 1. Verificar la GPU"),
        code(
            '''#@title 1. Verificar GPU y versiones
import subprocess

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

import torch
print("PyTorch:", torch.__version__)
print("CUDA de PyTorch:", torch.version.cuda)
print("GPU disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("SIN GPU: ve a Entorno de ejecucion > Cambiar tipo de entorno > T4 GPU")'''
        ),
        md(
            """## 2. Dependencias

Colab ya trae `torch` y casi siempre `ultralytics`. **No se reinstala nada por
gusto**: instalar `ultralytics` a mano arrastra otra versión de `torch` y puede
romper la GPU de la sesión. La celda sólo instala si falta el paquete."""
        ),
        code(
            '''#@title 2. Comprobar ultralytics sin tocar torch
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("ultralytics") is None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3,<9"],
        check=True,
    )

import ultralytics
print("ultralytics", ultralytics.__version__)
ultralytics.checks()'''
        ),
        md(
            """## 3. Traer el paquete de datos

Sube `detector_v2_clean.zip` a `Mi unidad/ajedrez_v2/` y deja `USAR_DRIVE = True`.
El zip se descomprime en `/content` (disco local de la VM), **no** en Drive:
leer miles de imágenes montadas desde Drive es mucho más lento."""
        ),
        code(
            '''#@title 3. Descomprimir detector_v2_clean.zip y contar por split
import shutil
import zipfile
from pathlib import Path

USAR_DRIVE = True
ZIP_EN_DRIVE = Path("/content/drive/MyDrive/ajedrez_v2/detector_v2_clean.zip")
DESTINO = Path("/content/ajedrez_v2/detector")
ESPERADO = {"train": 4372, "val": 73, "test": 64}

if USAR_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    origen = ZIP_EN_DRIVE
else:
    from google.colab import files
    subidos = files.upload()
    origen = Path(next(iter(subidos)))

if not origen.exists():
    raise FileNotFoundError(
        f"No encuentro {origen}. Sube detector_v2_clean.zip a "
        "Mi unidad/ajedrez_v2/ o pon USAR_DRIVE = False."
    )

if DESTINO.exists():
    shutil.rmtree(DESTINO)
DESTINO.mkdir(parents=True)

with zipfile.ZipFile(origen) as zf:
    zf.extractall(DESTINO)

print("Contenido:", sorted(p.name for p in DESTINO.iterdir()))
for split, esperado in ESPERADO.items():
    n_img = len(list((DESTINO / "images" / split).glob("*.jpg")))
    n_lbl = len(list((DESTINO / "labels" / split).glob("*.txt")))
    estado = "OK" if n_img == esperado == n_lbl else "REVISAR"
    print(f"  {split:5s} imagenes={n_img:5d} etiquetas={n_lbl:5d} "
          f"(esperado {esperado:5d}) {estado}")

assert (DESTINO / "data.yaml").exists(), "Falta data.yaml en el zip"

PROYECTO = Path("/content/drive/MyDrive/ajedrez_v2/detector")
PROYECTO.mkdir(parents=True, exist_ok=True)
print("Registro de entrenamientos:", PROYECTO)'''
        ),
        code(
            '''#@title 3b. Corregir `path` del data.yaml (REQUERIDO por Ultralytics)
# Sin esto, Ultralytics resuelve `path` contra su datasets_dir GLOBAL
# (/content/datasets) y falla con: "missing path '/content/images/val'".
import re

RUTA_YAML = DESTINO / "data.yaml"
texto = RUTA_YAML.read_text(encoding="utf-8")
texto, cambios = re.subn(r"^path: .*$", f"path: {DESTINO.as_posix()}",
                         texto, flags=re.M)
RUTA_YAML.write_text(texto, encoding="utf-8")
print(f"path reescrito en {RUTA_YAML} ({cambios} cambio(s))")
assert cambios >= 1, "data.yaml no tenia linea 'path:'"
print("\\n".join(texto.splitlines()[:6]))'''
        ),
        md(
            """## 4. Auditar el dataset antes de entrenar

Atrapa los errores silenciosos: imagen y etiqueta con el mismo nombre en los
tres splits, líneas YOLO válidas con clases en rango, `data.yaml` con las 12
clases en el orden de `classes.txt` y ninguna imagen repetida entre splits."""
        ),
        code(
            '''#@title 4. Auditoria del dataset
from collections import Counter

import yaml

CLASES_ESPERADAS = [
    "peon_b", "torre_b", "caballo_b", "alfil_b", "dama_b", "rey_b",
    "peon_n", "torre_n", "caballo_n", "alfil_n", "dama_n", "rey_n",
]
RAIZ = DESTINO
cfg = yaml.safe_load((RAIZ / "data.yaml").read_text(encoding="utf-8"))
NOMBRES = [cfg["names"][i] for i in range(cfg["nc"])]
print("Suites:", cfg["train"], "|", cfg["val"], "|", cfg["test"])
print("Clases:", NOMBRES)
if NOMBRES != CLASES_ESPERADAS:
    raise SystemExit("data.yaml no respeta el orden de classes.txt: NO entrenes asi")

problemas = []
resumen = {}
vistos = {}

for split in ("train", "val", "test"):
    dir_img = RAIZ / "images" / split
    dir_lbl = RAIZ / "labels" / split
    imgs = {p.stem for p in dir_img.glob("*.jpg")}
    lbls = {p.stem for p in dir_lbl.glob("*.txt")}
    if imgs - lbls:
        problemas.append(f"{split}: {len(imgs - lbls)} imagenes sin etiqueta")
    if lbls - imgs:
        problemas.append(f"{split}: {len(lbls - imgs)} etiquetas sin imagen")

    conteo = Counter()
    vacias = 0
    invalidas = 0
    for txt in dir_lbl.glob("*.txt"):
        lineas = [l for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lineas:
            vacias += 1
        for linea in lineas:
            partes = linea.split()
            if len(partes) != 5:
                invalidas += 1
                continue
            clase = int(partes[0])
            x, y, w, h = (float(v) for v in partes[1:])
            if not (0 <= clase < 12) or not all(0.0 < v <= 1.0 for v in (x, y, w, h)):
                invalidas += 1
                continue
            conteo[clase] += 1
    vistos[split] = imgs
    resumen[split] = {"imagenes": len(imgs), "cajas": sum(conteo.values()),
                      "conteo": conteo}
    print(f"{split:5s} imagenes={len(imgs):5d} cajas={sum(conteo.values()):6d} "
          f"negativos={vacias:3d} invalidas={invalidas}")
    if invalidas:
        problemas.append(f"{split}: {invalidas} lineas invalidas")

for a in ("train", "val", "test"):
    for b in ("train", "val", "test"):
        if a < b and vistos[a] & vistos[b]:
            problemas.append(f"{len(vistos[a] & vistos[b])} imagenes repetidas "
                             f"entre {a} y {b}")

for split in ("val", "test"):
    faltan = [NOMBRES[i] for i in range(12) if resumen[split]["conteo"][i] == 0]
    if faltan:
        print(f"AVISO {split}: clases sin ejemplos -> {faltan}")

print()
if problemas:
    print("PROBLEMAS:")
    for p in problemas:
        print("  -", p)
    raise SystemExit("Corrige el dataset antes de entrenar.")
print("Dataset OK. Puedes entrenar.")'''
        ),
        md(
            """## 5. Configuración del entrenamiento

`MODO = "desde_cero"`: se parte de **`yolo11n.pt`** (pesos COCO) y se entrena
todo el modelo con `lr0 = 0.01`. Desde cero necesita más learning rate que el
fine-tune anterior (0.001) porque no hay nada que preservar.

Aumentos pensados para un tablero, no para sprites:

- `fliplr = 0.0`, `flipud = 0.0`: el tablero tiene orientación **fija** y de ella
  depende el orden de clases (a1 abajo-izquierda). Voltear convierte una torre
  blanca en una negra de otra esquina.
- `mosaic = 0.5` y `close_mosaic = 15`: mosaico suave y las últimas 15 épocas con
  imágenes completas.
- `cache = "ram"`: 4,044 imágenes caben en la RAM de Colab.
- `seed = 42`, `deterministic = True`: misma corrida, mismo resultado.

El proyecto se escribe en **Drive** para no perder `best.pt` si la sesión cae."""
        ),
        code(
            '''#@title 5. Configuracion
MODO = "desde_cero"     # "desde_cero" (yolo11n.pt) | "fino" (reanudar un .pt)
BASE = "yolo11n.pt"     # MODO="desde_cero": pesos COCO, NO el modelo viejo
EPOCAS = 120
BATCH = 16              # YOLO11n a 640 px entra comodo en una T4
IMGSZ = 640
LR0 = 0.01              # desde cero; el fine-tune usaba 0.001
PATIENCE = 25
NOMBRE_RUN = "desde_cero_v2_clean"

DATA = DESTINO / "data.yaml"
assert DATA.exists(), "Falta el paquete: vuelve a la celda 3"

print("Modo:", MODO)
print("Base:", BASE)
print("Epocas:", EPOCAS, "| batch:", BATCH, "| imgsz:", IMGSZ, "| lr0:", LR0)
print("Salida:", PROYECTO / NOMBRE_RUN)'''
        ),
        code(
            '''#@title 6. Entrenar
from pathlib import Path

from ultralytics import YOLO

COMUNES = dict(
    data=str(DATA),
    imgsz=IMGSZ,
    batch=BATCH,
    workers=2,
    cache="ram",
    seed=42,
    deterministic=True,
    fliplr=0.0,
    flipud=0.0,
    mosaic=0.5,
    close_mosaic=15,
    cos_lr=True,
    patience=PATIENCE,
    plots=True,
    val=True,
    project=str(PROYECTO),
    name=NOMBRE_RUN,
    exist_ok=True,
    save_period=5,
)

modelo = YOLO(BASE if MODO == "desde_cero" else MODO)
resultado = modelo.train(epochs=EPOCAS, lr0=LR0, lrf=0.01, **COMUNES)

PESO_FINAL = Path(resultado.save_dir) / "weights" / "best.pt"
print("Modelo final:", PESO_FINAL, "| existe:", PESO_FINAL.exists())'''
        ),
        md(
            """## 6. Evaluar sobre `test` (fotos reales, nunca vistas)

Este es el número que vale. `valid` decide cuándo parar (lo usa `patience`);
`test` es la medida final.

**No hay línea base comparable.** El modelo del lote anterior daba `mAP50 0.546`
pero se entrenó con etiquetas cruzadas caballo/alfil y sobre otro test: ese
número **no** es una referencia honesta para este dataset. La comparación válida
es contra la primera época de esta misma corrida (está en `results.csv`)."""
        ),
        code(
            '''#@title 7. Evaluar en test + metricas por clase
from ultralytics import YOLO

modelo_eval = YOLO(str(PESO_FINAL))
metricas = modelo_eval.val(data=str(DATA), split="test", imgsz=IMGSZ,
                           batch=BATCH, plots=True, project=str(PROYECTO),
                           name="eval_test", exist_ok=True)

print()
print(f"TEST  mAP50   : {metricas.box.map50:.4f}")
print(f"TEST  mAP50-95: {metricas.box.map:.4f}")
print(f"TEST  precision: {metricas.box.mp:.4f}")
print(f"TEST  recall   : {metricas.box.mr:.4f}")

print()
print(f"{'clase':12s} {'P':>7s} {'R':>7s} {'mAP50-95':>9s}")
for i, nombre in metricas.names.items():
    p = metricas.box.p[i] if i < len(metricas.box.p) else 0.0
    r = metricas.box.r[i] if i < len(metricas.box.r) else 0.0
    m = metricas.box.maps[i] if i < len(metricas.box.maps) else 0.0
    print(f"{nombre:12s} {p:7.3f} {r:7.3f} {m:9.3f}")'''
        ),
        md(
            """## 7. Confusiones reales -> predichas

La matriz de confusión dice qué clases se confunden entre sí. El par a vigilar
es `caballo_n <-> alfil_n`: fue el que se cruzó en las etiquetas del lote
anterior. Si aquí aparece con fuerza, el problema es de datos, no de épocas."""
        ),
        code(
            '''#@title 8. Confusiones mas frecuentes
import numpy as np

cm = getattr(getattr(metricas, "confusion_matrix", None), "matrix", None)
if cm is None:
    print("Esta version de Ultralytics no expuso la matriz de confusion.")
else:
    cm = np.asarray(cm)
    nombres = metricas.names
    pares = []
    for real in range(min(len(nombres), cm.shape[0])):
        for pred in range(min(len(nombres), cm.shape[1])):
            if real != pred and cm[real, pred] > 0:
                pares.append((float(cm[real, pred]), nombres[real], nombres[pred]))
    pares.sort(reverse=True)
    print("Confusiones mas frecuentes (real -> predicha):")
    if not pares:
        print("  Ninguna. Excelente.")
    for cantidad, real, pred in pares[:12]:
        print(f"  {real:12s} -> {pred:12s} {cantidad:g}")'''
        ),
        md(
            """## 8. Guardar todo en Drive

Sin esta celda, si Colab cierra la sesión se pierde el entrenamiento. `last.pt`
permite reanudar; `best.pt` es el modelo que usa el pipeline."""
        ),
        code(
            '''#@title 9. Copiar modelo y curvas a Drive
import shutil

SALIDA = PROYECTO / "entrega"
SALIDA.mkdir(parents=True, exist_ok=True)

shutil.copy2(PESO_FINAL, SALIDA / "best.pt")

ultimo = PESO_FINAL.parent / "last.pt"
if ultimo.exists():
    shutil.copy2(ultimo, SALIDA / "last.pt")

run_dir = PESO_FINAL.parent.parent
for patron in ("results.csv", "results.png", "confusion_matrix.png",
               "confusion_matrix_normalized.png", "PR_curve.png", "F1_curve.png"):
    for archivo in run_dir.glob(patron):
        shutil.copy2(archivo, SALIDA / archivo.name)

shutil.copy2(DATA, SALIDA / "data.yaml")

print("Guardado en", SALIDA)
for p in sorted(SALIDA.iterdir()):
    print(f"  {p.name:38s} {p.stat().st_size / 1024:8.1f} KB")'''
        ),
        md("## 9. Prueba visual rápida"),
        code(
            '''#@title 10. Predecir sobre una foto de test
import random

from IPython.display import display
from PIL import Image

fotos = sorted((DESTINO / "images" / "test").glob("*.jpg"))
elegida = random.choice(fotos)
print("Foto:", elegida.name)

prediccion = modelo_eval.predict(str(elegida), imgsz=IMGSZ, conf=0.40, verbose=False)[0]
salida = Path("/content/prediccion_v2_clean.jpg")
Image.fromarray(prediccion.plot()[..., ::-1]).save(salida)
display(Image.open(salida))
print(f"{len(prediccion.boxes)} detecciones")'''
        ),
        md(
            """## Si algo falla

| Síntoma | Causa | Solución |
|---|---|---|
| `SIN GPU` | Entorno sin GPU | Entorno de ejecución > Cambiar tipo de entorno > T4 GPU |
| `No encuentro ...detector_v2_clean.zip` | Ruta de Drive distinta | Revisa `ZIP_EN_DRIVE` o pon `USAR_DRIVE = False` |
| `data.yaml no respeta el orden` | Zip editado a mano | Vuelve a generarlo con `scripts/preparar_colab_v2_clean.py` |
| `CUDA out of memory` | `BATCH` alto | Baja `BATCH` a 8 |
| mAP50 alto en valid y bajo en test | Sobreajuste a val | Baja `EPOCAS`, sube `PATIENCE` |
| Se perdió el entrenamiento | Colab cerró la sesión | Reanuda con `YOLO(...last.pt).train(resume=True)` |"""
        ),
    ]
    return {"cells": celdas, "metadata": metadatos(
        "ajedrez_v2_detector_desde_cero.ipynb"), "nbformat": 4, "nbformat_minor": 0}


# ===========================================================================
# CLASIFICADOR
# ===========================================================================
def notebook_clasificador() -> dict:
    celdas = [
        md(
            """# Clasificador por casilla v2 — DESDE CERO (dataset limpio)

Entrena **(b) el clasificador de 13 clases**: las 12 piezas + `vacia`.

| | |
|---|---|
| Modelo | `mobilenet_v3_small`, `weights=IMAGENET`, **sin congelar** |
| Entrada | casilla rectificada 100x100 -> 128x128 |
| Paquete | `clasificador_v2_clean.zip` (~474 MB) en `Mi unidad/ajedrez_v2/` |
| Train | 139,200 casillas de 2,175 tableros |
| Valid | 17,408 casillas de 272 tableros |
| Test | 17,408 casillas de 272 tableros |
| Épocas | 20 con `AMP` y pesos de clase por frecuencia inversa |

**Por qué este modelo:** YOLO detecta cajas e ignora que el tablero es una rejilla
conocida de 64 casillas. Aquí se rectifica el tablero, se recortan las 64 casillas
y se clasifica cada una. El reparto se hace **por tablero, nunca por casilla**:
las 64 casillas de una misma foto caen siempre en el mismo split.

Los recortes reales del `valid` y del `test` son **los mismos** que usa el
detector (mismo reparto por imagen fuente): la comparación de T7 se mide sobre
las mismas fotos.

Antes de ejecutar: **Entorno de ejecución > Cambiar tipo de entorno > T4 GPU**."""
        ),
        md("## 1. Verificar la GPU"),
        code(
            '''#@title 1. Verificar GPU y versiones
import subprocess

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

import torch
print("PyTorch:", torch.__version__)
print("CUDA de PyTorch:", torch.version.cuda)
print("GPU disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("SIN GPU: ve a Entorno de ejecucion > Cambiar tipo de entorno > T4 GPU")'''
        ),
        md(
            """## 2. Dependencias

`torch`, `torchvision` y `Pillow` ya vienen en Colab. La celda sólo instala si
falta algo y **nunca** reinstala `torch`: cambiarlo rompe la GPU de la sesión."""
        ),
        code(
            '''#@title 2. Comprobar dependencias sin tocar torch
import importlib.util
import subprocess
import sys

for paquete, modulo in (("torchvision", "torchvision"), ("pillow", "PIL")):
    if importlib.util.find_spec(modulo) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", paquete],
                       check=True)

import torch
import torchvision
print("torch", torch.__version__, "| torchvision", torchvision.__version__)'''
        ),
        md(
            """## 3. Traer el paquete de datos

El zip trae:

```
casillas/<clase>/<archivo>.jpg        191,072 recortes (13 clases)
                                      100x100 compuesto/reales + 128x128 aumentadas
split/{train,valid,test}.txt          una ruta RELATIVA por linea
labels/clases.txt                     las 13 clases en orden
MANIFEST.txt                          de donde sale cada casilla
```

La lista de cada split es la que manda: no hay `ImageFolder` ni carpetas por
split. Se descomprime en `/content` (disco local), no en Drive."""
        ),
        code(
            '''#@title 3. Descomprimir clasificador_v2_clean.zip y contar por split
import shutil
import zipfile
from pathlib import Path

USAR_DRIVE = True
ZIP_EN_DRIVE = Path("/content/drive/MyDrive/ajedrez_v2/clasificador_v2_clean.zip")
DESTINO = Path("/content/ajedrez_v2/clasificador")
ESPERADO = {"train": 157280, "valid": 16896, "test": 16896}

if USAR_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    origen = ZIP_EN_DRIVE
else:
    from google.colab import files
    subidos = files.upload()
    origen = Path(next(iter(subidos)))

if not origen.exists():
    raise FileNotFoundError(
        f"No encuentro {origen}. Sube clasificador_v2_clean.zip a "
        "Mi unidad/ajedrez_v2/ o pon USAR_DRIVE = False."
    )

if DESTINO.exists():
    shutil.rmtree(DESTINO)
DESTINO.mkdir(parents=True)

with zipfile.ZipFile(origen) as zf:
    zf.extractall(DESTINO)

print("Contenido:", sorted(p.name for p in DESTINO.iterdir()))
for split, esperado in ESPERADO.items():
    n = sum(1 for linea in (DESTINO / "split" / f"{split}.txt").read_text().splitlines()
            if linea.strip())
    print(f"  {split:5s} rutas={n:6d} (esperado {esperado:6d}) "
          f"{'OK' if n == esperado else 'REVISAR'}")

CLASES = [l.strip() for l in (DESTINO / "labels" / "clases.txt").read_text().splitlines()
          if l.strip()]
print("Clases (13):", CLASES)
assert len(CLASES) == 13 and CLASES[12] == "vacia", "labels/clases.txt no tiene 13 clases"

PROYECTO = Path("/content/drive/MyDrive/ajedrez_v2/clasificador")
PROYECTO.mkdir(parents=True, exist_ok=True)
print("Salida:", PROYECTO)'''
        ),
        md(
            """## 4. Dataset, transformaciones y pesos de clase

`Dataset` mínimo que lee la lista de rutas del `.txt` y saca la clase del
**nombre de la carpeta** (`casillas/<clase>/archivo.jpg`).

- `Resize(128)` + normalización de ImageNet (el modelo viene de ahí).
- Sólo en train, `ColorJitter` suave y `RandomRotation` de ±12°: las fichas
  físicas se pueden colocar giradas.
- **Pesos de clase** = frecuencia inversa sobre el split de train. `vacia` es el
  55.9 % de las casillas; sin el peso el modelo aprende a decir "vacia" y listo."""
        ),
        code(
            '''#@title 4. Datasets, loaders y pesos de clase
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

RAIZ = DESTINO
TAMANO = 128
BATCH = 128
EPOCAS = 20
LR = 1e-3
SEMILLA = 42
NUM_WORKERS = 4

torch.manual_seed(SEMILLA)

TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize((TAMANO, TAMANO)),
    transforms.RandomRotation(12),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
TRANSFORM_EVAL = transforms.Compose([
    transforms.Resize((TAMANO, TAMANO)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class Casillas(Dataset):
    """Casillas de un split, leidas desde su .txt (rutas relativas al zip)."""

    def __init__(self, raiz: Path, txt: Path, clases: list, transform):
        self.raiz = raiz
        self.clases = clases
        self.transform = transform
        self.rutas = [l.strip() for l in txt.read_text(encoding="utf-8").splitlines()
                      if l.strip()]

    def __len__(self) -> int:
        return len(self.rutas)

    def __getitem__(self, indice: int):
        relativa = self.rutas[indice]
        clase = relativa.split("/")[1]          # casillas/<clase>/<archivo>.jpg
        imagen = Image.open(self.raiz / relativa).convert("RGB")
        return self.transform(imagen), self.clases.index(clase)


conjuntos = {
    split: Casillas(RAIZ, RAIZ / "split" / f"{split}.txt", CLASES,
                    TRANSFORM_TRAIN if split == "train" else TRANSFORM_EVAL)
    for split in ("train", "valid", "test")
}
cargadores = {
    "train": DataLoader(conjuntos["train"], batch_size=BATCH, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True),
    "valid": DataLoader(conjuntos["valid"], batch_size=BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True),
    "test": DataLoader(conjuntos["test"], batch_size=BATCH, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True),
}

conteo_train = Counter(ruta.split("/")[1] for ruta in conjuntos["train"].rutas)
faltan = [c for c in CLASES if conteo_train[c] == 0]
if faltan:
    raise SystemExit(f"Clases sin ejemplos en train: {faltan}. Revisa split/train.txt.")

totales = sum(conteo_train.values())
pesos = torch.tensor(
    [totales / (len(CLASES) * conteo_train[c]) for c in CLASES], dtype=torch.float32
)
pesos = pesos / pesos.mean()   # media 1: no mueve la escala del learning rate

print(f"{'clase':12s} {'train':>8s} {'%':>7s} {'peso':>6s}")
for clase, peso in zip(CLASES, pesos.tolist()):
    n = conteo_train[clase]
    print(f"{clase:12s} {n:8d} {100 * n / totales:6.2f}% {peso:6.2f}")
print(f"\\nvacia = {100 * conteo_train['vacia'] / totales:.2f}% del train")
print("items:", {k: len(v) for k, v in conjuntos.items()})'''
        ),
        md(
            """## 5. Modelo, optimizador y función de pérdida

`mobilenet_v3_small` con `weights=IMAGENET` y **todas** las capas entrenables:
no se congela `features`, porque el dominio (fichas sobre tablero impreso) no se
parece a ImageNet. `AdamW(lr=1e-3)` + cosine, `CrossEntropyLoss(weight=...)` y
**AMP** (`torch.amp.autocast` + `GradScaler`): en una T4 casi duplica el
rendimiento por época sin cambiar el resultado."""
        ),
        code(
            '''#@title 5. Crear el modelo (ImageNet, sin congelar) y AMP
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
USAR_AMP = DISPOSITIVO == "cuda"


def crear_modelo(n_clases: int) -> nn.Module:
    try:
        red = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
    except (AttributeError, TypeError):          # torchvision viejo
        red = models.mobilenet_v3_small(pretrained=True)
    for parametro in red.features.parameters():  # SIN congelar
        parametro.requires_grad = True
    entrada = red.classifier[3].in_features
    red.classifier[3] = nn.Linear(entrada, n_clases)
    return red


modelo = crear_modelo(len(CLASES)).to(DISPOSITIVO)
optimizador = AdamW(modelo.parameters(), lr=LR, weight_decay=1e-4)
planificador = CosineAnnealingLR(optimizador, T_max=EPOCAS)
criterio = nn.CrossEntropyLoss(weight=pesos.to(DISPOSITIVO))
escalador = torch.amp.GradScaler("cuda", enabled=USAR_AMP)

n_parametros = sum(p.numel() for p in modelo.parameters())
entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
print("Dispositivo:", DISPOSITIVO, "| AMP:", USAR_AMP)
print(f"Parametros: {n_parametros / 1e6:.2f} M ({entrenables / 1e6:.2f} M entrenables)")
print("Salida:", modelo.classifier[3].out_features, "clases")'''
        ),
        md(
            """## 6. Entrenar (20 épocas)

Se guarda en Drive cada vez que `valid` mejora. Si Colab se desconecta, el
`best_clasificador.pt` sigue en tu unidad."""
        ),
        code(
            '''#@title 6. Entrenar con AMP
import copy
import time

def evaluar(red, cargador):
    red.eval()
    perdida_total = 0.0
    aciertos = 0
    vistos = 0
    with torch.no_grad():
        for imagenes, etiquetas in cargador:
            imagenes = imagenes.to(DISPOSITIVO, non_blocking=True)
            etiquetas = etiquetas.to(DISPOSITIVO, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=USAR_AMP):
                salida = red(imagenes)
                perdida = criterio(salida, etiquetas)
            perdida_total += perdida.item() * len(etiquetas)
            aciertos += (salida.argmax(1) == etiquetas).sum().item()
            vistos += len(etiquetas)
    return perdida_total / max(vistos, 1), aciertos / max(vistos, 1)


SALIDA = PROYECTO
SALIDA.mkdir(parents=True, exist_ok=True)
mejor_acc = 0.0
historial = []

for epoca in range(1, EPOCAS + 1):
    arranque = time.time()
    modelo.train()
    perdida_total = 0.0
    aciertos = 0
    vistos = 0
    for imagenes, etiquetas in cargadores["train"]:
        imagenes = imagenes.to(DISPOSITIVO, non_blocking=True)
        etiquetas = etiquetas.to(DISPOSITIVO, non_blocking=True)
        optimizador.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=USAR_AMP):
            salida = modelo(imagenes)
            perdida = criterio(salida, etiquetas)
        escalador.scale(perdida).backward()
        escalador.step(optimizador)
        escalador.update()
        perdida_total += perdida.item() * len(etiquetas)
        aciertos += (salida.argmax(1) == etiquetas).sum().item()
        vistos += len(etiquetas)
    planificador.step()

    perdida_train = perdida_total / max(vistos, 1)
    acc_train = aciertos / max(vistos, 1)
    perdida_val, acc_val = evaluar(modelo, cargadores["valid"])
    segundos = time.time() - arranque
    historial.append((epoca, perdida_train, acc_train, perdida_val, acc_val))
    print(f"epoca {epoca:2d}/{EPOCAS}  loss={perdida_train:.4f} "
          f"train={acc_train:.3f} valid={acc_val:.3f}  ({segundos:.0f}s)")

    if acc_val > mejor_acc:
        mejor_acc = acc_val
        torch.save({
            "modelo": copy.deepcopy(modelo.state_dict()),
            "clases": CLASES,
            "epoca": epoca,
            "acc_valid": acc_val,
            "tamano": TAMANO,
        }, SALIDA / "best_clasificador.pt")
        print(f"  -> mejor hasta ahora (valid={acc_val:.3f}), guardado en Drive")

print()
print(f"Mejor accuracy en valid: {mejor_acc:.4f}")
print("Pesos:", SALIDA / "best_clasificador.pt")'''
        ),
        md(
            """## 7. Matriz de confusión por clase

El accuracy global esconde el problema: `vacia` es el 56 % de las casillas. Si el
modelo sólo acierta `vacia`, el número se ve bien y el pipeline falla. Aquí se ve
clase por clase, y el par `caballo_n <-> alfil_n` en particular."""
        ),
        code(
            '''#@title 7. Matriz de confusion y reporte por clase
import numpy as np

def matriz_confusion(red, cargador, n_clases: int) -> np.ndarray:
    matriz = np.zeros((n_clases, n_clases), dtype=np.int64)
    red.eval()
    with torch.no_grad():
        for imagenes, etiquetas in cargador:
            salida = red(imagenes.to(DISPOSITIVO)).argmax(1).cpu().numpy()
            for real, pred in zip(etiquetas.numpy(), salida):
                matriz[real, pred] += 1
    return matriz


if not (SALIDA / "best_clasificador.pt").exists():
    raise SystemExit("No hay best_clasificador.pt: ejecuta la celda 6 primero")

pesos_guardados = torch.load(SALIDA / "best_clasificador.pt", map_location=DISPOSITIVO)
modelo.load_state_dict(pesos_guardados["modelo"])

for split in ("valid", "test"):
    matriz = matriz_confusion(modelo, cargadores[split], len(CLASES))
    total = matriz.sum()
    aciertos = np.trace(matriz)
    print(f"== {split} ==")
    print(f"   accuracy global: {aciertos / total:.4f}  ({aciertos}/{total})")
    print(f"   {'clase':12s} {'n':>7s} {'prec':>6s} {'recall':>6s}")
    for i, clase in enumerate(CLASES):
        n = matriz[i].sum()
        if n == 0:
            print(f"   {clase:12s} {0:7d}   (sin ejemplos)")
            continue
        recall = matriz[i, i] / n
        predichos = matriz[:, i].sum()
        precision = matriz[i, i] / predichos if predichos else 0.0
        print(f"   {clase:12s} {n:7d} {precision:6.3f} {recall:6.3f}")

    errores = []
    for real in range(len(CLASES)):
        for pred in range(len(CLASES)):
            if real != pred and matriz[real, pred] > 0:
                errores.append((int(matriz[real, pred]), CLASES[real], CLASES[pred]))
    errores.sort(reverse=True)
    print("   confusiones mas frecuentes:")
    for cantidad, real, pred in errores[:8]:
        print(f"     {real:12s} -> {pred:12s} {cantidad}")
    print()'''
        ),
        md(
            """## 8. Guardar el modelo final en Drive

`best_clasificador.pt` guarda el `state_dict`, la lista de clases y el tamaño de
entrada. Para usarlo hay que reconstruir `mobilenet_v3_small` y cargar el
`state_dict`: el archivo no trae la arquitectura."""
        ),
        code(
            '''#@title 8. Verificar que el modelo se puede recargar y listar Drive
import shutil

SALIDA_ENTREGA = PROYECTO / "entrega"
SALIDA_ENTREGA.mkdir(parents=True, exist_ok=True)

shutil.copy2(PROYECTO / "best_clasificador.pt",
             SALIDA_ENTREGA / "best_clasificador.pt")
(SALIDA_ENTREGA / "clases.txt").write_text("\\n".join(CLASES) + "\\n",
                                           encoding="utf-8")

comprobacion = crear_modelo(len(CLASES)).to(DISPOSITIVO)
comprobacion.load_state_dict(torch.load(SALIDA_ENTREGA / "best_clasificador.pt",
                                        map_location=DISPOSITIVO)["modelo"])
comprobacion.eval()
print("Recarga OK:", SALIDA_ENTREGA / "best_clasificador.pt")
for p in sorted(SALIDA_ENTREGA.iterdir()):
    print(f"  {p.name:28s} {p.stat().st_size / 1024:8.1f} KB")'''
        ),
        md(
            """## Cómo entra esto en el pipeline v2

1. **Detector YOLO** (`best.pt` del otro notebook) — segunda opinión.
2. **Rectificación** del tablero con las 4 esquinas ArUco -> 800x800.
3. **Clasificador por casilla** (`best_clasificador.pt`): 64 recortes, 13 clases.
4. **Filtro de legalidad** con `python-chess` sobre el FEN construido.

## Si algo falla

| Síntoma | Causa | Solución |
|---|---|---|
| `SIN GPU` | Entorno sin GPU | Entorno de ejecución > Cambiar tipo de entorno > T4 GPU |
| `No encuentro ...clasificador_v2_clean.zip` | Ruta de Drive distinta | Revisa `ZIP_EN_DRIVE` o pon `USAR_DRIVE = False` |
| Descomprimir tarda | 174 mil archivos pequeños | Es normal: 4-6 minutos |
| `CUDA out of memory` | `BATCH` alto | Baja `BATCH` a 64 |
| `accuracy` alta pero el pipeline falla | El modelo dice `vacia` a todo | Mira el recall por clase |
| DataLoader lento | `NUM_WORKERS` bajo | Sube `NUM_WORKERS` a 8 |"""
        ),
    ]
    return {"cells": celdas, "metadata": metadatos(
        "ajedrez_v2_clasificador_desde_cero.ipynb"), "nbformat": 4, "nbformat_minor": 0}


def main() -> int:
    salidas = {
        "ajedrez_v2_detector_desde_cero.ipynb": notebook_detector(),
        "ajedrez_v2_clasificador_desde_cero.ipynb": notebook_clasificador(),
    }
    for nombre, contenido in salidas.items():
        ruta = AQUI / nombre
        ruta.write_text(json.dumps(contenido, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        print(f"escrito {ruta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
