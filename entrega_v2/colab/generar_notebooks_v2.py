#!/usr/bin/env python
"""Genera los dos notebooks de Colab de la arquitectura v2.

Mismo patron que `ajedrez-yolo/generar_notebook_colab.py`: los notebooks se
escriben desde aqui para que las celdas se puedan revisar y corregir en un solo
sitio, sin editar JSON a mano.

    python colab/generar_notebooks_v2.py

Escribe, junto a este archivo:
    ajedrez_v2_detector.ipynb
    ajedrez_v2_clasificador.ipynb
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


def notebook(celdas: list[dict], nombre: str) -> dict:
    return {
        "cells": celdas,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": nombre, "provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


# ==========================================================================
# NOTEBOOK 1 - DETECTOR YOLO
# ==========================================================================
DETECTOR = [
    md("""# Detector de piezas v2 — entrenamiento en Colab

Este notebook entrena **(a) el detector YOLO** de la arquitectura v2.

| | |
|---|---|
| Modelo base | `modelo_sintetico.pt` (YOLO11n de v1, **mismo orden de 12 clases**) |
| Paquete | `detector_v2.zip` (272.6 MB) en `Mi unidad/ajedrez_v2/` |
| Train | 3,802 tableros generados (1,544 sinteticos + 2,258 compuestos sobre fondo real) |
| Valid | 84 fotos reales: 12 recortes x 7 variantes |
| Test | 56 fotos reales: 8 recortes x 7 variantes |
| Epocas | 60 (con `patience=15`) |

**La regla que no se toca:** `valid` y `test` son SOLO fotos reales de la camara.
El train es todo lo generado. Si val o test tuvieran imagenes sinteticas, la
metrica mediria lo que el modelo ya vio y no serviria para decidir nada.

Antes de ejecutar: **Entorno de ejecucion > Cambiar tipo de entorno > T4 GPU**.
Despues corre las celdas de arriba a abajo. Los pesos se guardan en tu Drive, asi
que si Colab se desconecta no pierdes el entrenamiento."""),
    md("## 1. Verificar la GPU"),
    code("""#@title 1. Verificar GPU y versiones
!nvidia-smi

import torch
print("PyTorch:", torch.__version__)
print("CUDA de PyTorch:", torch.version.cuda)
print("GPU disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("SIN GPU: ve a Entorno de ejecucion > Cambiar tipo de entorno > T4 GPU")"""),
    md("""## 2. Dependencias

Colab ya trae `torch`, `torchvision` y (casi siempre) `ultralytics`. **No se
reinstala nada por gusto**: instalar `ultralytics` a mano arrastra otra version
de `torch` y puede romper la GPU de la sesion. La celda solo instala si el
paquete no existe."""),
    code("""#@title 2. Comprobar ultralytics sin tocar torch
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
ultralytics.checks()"""),
    md("""## 3. Traer el paquete de datos

Dos opciones:

- **Drive (recomendado):** sube `detector_v2.zip` a `Mi unidad/ajedrez_v2/` y
  deja `USAR_DRIVE = True`.
- **Subida directa:** pon `USAR_DRIVE = False` y elige el archivo cuando lo pida.
  Ojo: `files.upload()` de Colab no aguanta bien archivos de cientos de MB; para
  273 MB usa Drive.

El zip se descomprime en `/content` (disco local de la VM) y **no** en Drive:
leer miles de imagenes pequeñas montadas desde Drive es mucho mas lento."""),
    code("""#@title 3. Descomprimir detector_v2.zip y contar por split
import shutil
import zipfile
from pathlib import Path

USAR_DRIVE = True
ZIP_EN_DRIVE = Path("/content/drive/MyDrive/ajedrez_v2/detector_v2.zip")
DESTINO = Path("/content/ajedrez_v2/detector")
ESPERADO = {"train": 3802, "val": 84, "test": 56}

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
        f"No encuentro {origen}. Sube detector_v2.zip a Mi unidad/ajedrez_v2/ "
        "o pon USAR_DRIVE = False."
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
assert (DESTINO / "weights" / "modelo_sintetico.pt").exists(), "Faltan los pesos base"

PROYECTO = Path("/content/drive/MyDrive/ajedrez_v2/detector")
PROYECTO.mkdir(parents=True, exist_ok=True)
print("Registro de entrenamientos:", PROYECTO)"""),
    code("""#@title 3b. Corregir `path` del data.yaml (REQUERIDO por Ultralytics)
# El zip trae `path` apuntando a la carpeta de descompresion de Colab. Si
# descomprimiste en otra ruta, esta celda lo reescribe con la correcta.
# Sin esto, Ultralytics resuelve `path` contra su datasets_dir GLOBAL
# (/content/datasets) y falla con: "missing path '/content/images/val'".
# Es el mismo error que costo una corrida entera en v1.
import re

RUTA_YAML = DESTINO / "data.yaml"
texto = RUTA_YAML.read_text(encoding="utf-8")
texto, cambios = re.subn(r"^path: .*$",
                         f"path: {DESTINO.as_posix()}", texto, flags=re.M)
RUTA_YAML.write_text(texto, encoding="utf-8")
print(f"path reescrito en {RUTA_YAML} ({cambios} cambio(s))")
print("\\n".join(texto.splitlines()[:6]))
assert cambios >= 1, "data.yaml no tenia linea 'path:'"          """),
    md("""## 4. Auditar el dataset antes de entrenar

Esta celda es la que atrapa los errores silenciosos. Comprueba:

- imagen y etiqueta con el mismo nombre en los tres splits,
- lineas YOLO validas y clases dentro de rango,
- `data.yaml` con las **12 clases en el orden de `classes.txt`** (si el orden
  esta cruzado, el modelo entrena bien y llama "caballo" a una torre),
- que ninguna imagen este en dos splits a la vez."""),
    code("""#@title 4. Auditoria del dataset
from collections import Counter
from pathlib import Path

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
    resumen[split] = {"imagenes": len(imgs), "cajas": sum(conteo.values()), "conteo": conteo}
    print(f"{split:5s} imagenes={len(imgs):5d} cajas={sum(conteo.values()):6d} "
          f"negativos={vacias:3d} invalidas={invalidas}")
    if invalidas:
        problemas.append(f"{split}: {invalidas} lineas invalidas")

for a in ("train", "val", "test"):
    for b in ("train", "val", "test"):
        if a < b and vistos[a] & vistos[b]:
            problemas.append(f"{len(vistos[a] & vistos[b])} imagenes repetidas entre {a} y {b}")

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
print("Dataset OK. Puedes entrenar.")"""),
    md("""## 5. Configuracion del entrenamiento

`MODO = "fino"`: parte de `modelo_sintetico.pt` y ajusta todo el modelo con
`lr0 = 0.001`. El modelo base ya distingue las piezas; lo que falta es adaptarlo
a la camara y a las fichas nuevas.

Aumentos pensados para un tablero, no para sprites:

- `fliplr = 0.0`, `flipud = 0.0`: el tablero tiene orientacion **fija** y de ella
  depende el sentido del orden de clases (a1 abajo-izquierda). Voltear la imagen
  convierte una torre blanca en una negra de otra esquina. Las rotaciones de
  **ficha** ya las trae el dataset (el compositor gira cada pieza 0-360).
- `mosaic = 0.5` y `close_mosaic = 10`: mosaico suave y las ultimas 10 epocas con
  imagenes completas, para que el modelo vea el tablero entero antes de parar.
- `cache = "ram"`: 3,802 imagenes caben en la RAM de Colab y quita la lectura de
  disco del camino critico.
- `seed = 42`, `deterministic = True`: misma corrida, mismo resultado.

El proyecto de entrenamiento se escribe **en Drive** para no perder `best.pt` si
la sesion se cae."""),
    code("""#@title 5. Configuracion
from pathlib import Path

MODO = "fino"          # "fino" (recomendado) | "dos_fases"
EPOCAS = 60
BATCH = 16             # YOLO11n a 640 px entra comodo en una T4
IMGSZ = 640
LR0 = 0.001
PATIENCE = 15
NOMBRE_RUN = "fino_v2"

BASE = DESTINO / "weights" / "modelo_sintetico.pt"
DATA = DESTINO / "data.yaml"
assert BASE.exists() and DATA.exists(), "Falta el paquete: vuelve a la celda 3"

print("Modo:", MODO)
print("Base:", BASE)
print("Epocas:", EPOCAS, "| batch:", BATCH, "| imgsz:", IMGSZ, "| lr0:", LR0)
print("Salida:", PROYECTO / NOMBRE_RUN)"""),
    code("""#@title 6. Entrenar
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
    close_mosaic=10,
    cos_lr=True,
    patience=PATIENCE,
    pretrained=True,
    plots=True,
    val=True,
    project=str(PROYECTO),
)

modelo = YOLO(str(BASE))

if MODO == "fino":
    resultado = modelo.train(
        epochs=EPOCAS,
        lr0=LR0,
        lrf=0.01,
        freeze=0,
        name=NOMBRE_RUN,
        exist_ok=True,
        save_period=5,
        **COMUNES,
    )
    PESO_FINAL = Path(resultado.save_dir) / "weights" / "best.pt"
else:
    fase1 = min(30, EPOCAS)
    modelo.train(epochs=fase1, lr0=0.01, freeze=10,
                 name=f"{NOMBRE_RUN}_fase1", exist_ok=True, save_period=5, **COMUNES)
    peso_fase1 = Path(modelo.trainer.save_dir) / "weights" / "best.pt"
    modelo = YOLO(str(peso_fase1))
    resultado = modelo.train(epochs=max(EPOCAS - fase1, 1), lr0=0.0005, lrf=0.01,
                             freeze=0, name=f"{NOMBRE_RUN}_fase2", exist_ok=True,
                             save_period=5, **COMUNES)
    PESO_FINAL = Path(resultado.save_dir) / "weights" / "best.pt"

print("Modelo final:", PESO_FINAL, "| existe:", PESO_FINAL.exists())"""),
    md("""## 6. Evaluar sobre `test` (fotos reales, nunca vistas)

Este es el numero que vale. `valid` sirve para decidir cuando parar (lo usa
`patience`); `test` es la medida final.

**Linea base medida con el paquete de este notebook:** `modelo_sintetico.pt`
**sin ajustar** da `mAP50 = 0.040`, `precision = 0.088`, `recall = 0.128` sobre
las 56 imagenes reales de `test` (medido en CPU con `scripts/verificar_detector_v2.py`).
Es un modelo de otro dominio: encuentra poco y dispara cajas donde no hay ficha.
Si el fine-tuning no sube `mAP50` muy por encima de 0.55 y la precision por
encima de 0.60, algo esta mal: revisa las etiquetas antes de tocar
hiperparametros."""),
    code("""#@title 7. Evaluar en test + metricas por clase
from ultralytics import YOLO

modelo_eval = YOLO(str(PESO_FINAL))
metricas = modelo_eval.val(data=str(DATA), split="test", imgsz=IMGSZ, batch=BATCH,
                           plots=True, project=str(PROYECTO), name="eval_test",
                           exist_ok=True)

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
    print(f"{nombre:12s} {p:7.3f} {r:7.3f} {m:9.3f}")"""),
    md("""## 7. Confusiones reales -> predichas

La matriz de confusion dice que clases se confunden entre si (por ejemplo
`torre_b -> dama_b`). Es lo que la metrica global no te cuenta."""),
    code("""#@title 8. Confusiones mas frecuentes
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
        print(f"  {real:12s} -> {pred:12s} {cantidad:g}")"""),
    md("""## 8. Guardar todo en Drive

Sin esta celda, si Colab cierra la sesion se pierde el entrenamiento. `last.pt`
permite reanudar; `best.pt` es el modelo que se usa despues."""),
    code("""#@title 9. Copiar modelo y curvas a Drive
import shutil
from pathlib import Path

SALIDA = PROYECTO / "entrega"
SALIDA.mkdir(parents=True, exist_ok=True)

shutil.copy2(PESO_FINAL, SALIDA / "best.pt")

ultimo = PESO_FINAL.parent / "last.pt"
if ultimo.exists():
    shutil.copy2(ultimo, SALIDA / "last.pt")

run_dir = PESO_FINAL.parent.parent
for patron in ("results.csv", "confusion_matrix.png", "confusion_matrix_normalized.png",
               "PR_curve.png", "F1_curve.png", "results.png"):
    for archivo in run_dir.glob(patron):
        shutil.copy2(archivo, SALIDA / archivo.name)

shutil.copy2(DATA, SALIDA / "data.yaml")

print("Guardado en", SALIDA)
for p in sorted(SALIDA.iterdir()):
    print(f"  {p.name:38s} {p.stat().st_size / 1024:8.1f} KB")"""),
    md("""## 9. Prueba visual rapida

Dibuja las predicciones sobre una foto real de `test`. Es lo que la matriz de
confusion no te cuenta: cajas dobles, fichas partidas, sombras detectadas."""),
    code("""#@title 10. Predecir sobre una foto de test
import random

from IPython.display import display
from PIL import Image

fotos = sorted((DESTINO / "images" / "test").glob("*.jpg"))
elegida = random.choice(fotos)
print("Foto:", elegida.name)

prediccion = modelo_eval.predict(str(elegida), imgsz=IMGSZ, conf=0.40, verbose=False)[0]
salida = Path("/content/prediccion_v2.jpg")
Image.fromarray(prediccion.plot()[..., ::-1]).save(salida)
display(Image.open(salida))
print(f"{len(prediccion.boxes)} detecciones")"""),
    md("""## Siguiente paso (no es este notebook)

Con `best.pt` ya entrenado se etiquetan automaticamente las **66 fotos reales de
medio juego** que hoy no tienen etiqueta (`dataset/reales/recortes/800x800/02_medio_juego`).
El plan esta en `colab/README.md`, seccion **"Siguiente paso"**: predecir con este
modelo, revisar a ojo las cajas dudosas, y volver a entrenar con ellas. Ese es el
camino para que `test` deje de ser solo posicion inicial y vacio.

Para comprobar el modelo en local (CPU, sin GPU) antes de usarlo:

```bash
python scripts/verificar_detector_v2.py --bundle dataset/colab_bundle/detector_v2.zip
```

## Si algo falla

| Sintoma | Causa | Solucion |
|---|---|---|
| `SIN GPU` | Entorno sin GPU | Entorno de ejecucion > Cambiar tipo de entorno > T4 GPU |
| `No encuentro ...detector_v2.zip` | Ruta de Drive distinta | Revisa `ZIP_EN_DRIVE` o pon `USAR_DRIVE = False` |
| `data.yaml no respeta el orden` | Zip editado a mano | Vuelve a descargar el `detector_v2.zip` del proyecto |
| `CUDA out of memory` | `BATCH` alto | Baja `BATCH` a 8 |
| mAP50 alto en valid y bajo en test | Sobreajuste a val | Baja `EPOCAS`, sube `PATIENCE`, revisa etiquetas de test |
| Se perdio el entrenamiento | Colab cerro la sesion | Reentrena con `resume` desde `PROYECTO/fino_v2/weights/last.pt` |"""),
]


# ==========================================================================
# NOTEBOOK 2 - CLASIFICADOR POR CASILLA
# ==========================================================================
CLASIFICADOR = [
    md("""# Clasificador por casilla v2 — entrenamiento en Colab

Este notebook entrena **(b) el clasificador de 13 clases** de la arquitectura v2:
las 12 piezas + `vacia`.

| | |
|---|---|
| Modelo | `mobilenet_v3_small` con `weights=DEFAULT` (ImageNet) |
| Entrada | casilla rectificada 100x100 -> redimensionada a 128x128 |
| Clases | 13: los 12 nombres de `classes.txt` + `vacia` (id 12) |
| Paquete | `clasificador_v2.zip` (406.7 MB) en `Mi unidad/ajedrez_v2/` |
| Train | 115,584 casillas de 1,806 tableros compuestos |
| Valid | 19,072 casillas (238 tableros: 226 compuestos + 12 recortes reales) |
| Test | 17,536 casillas (234 tableros: 226 compuestos + 8 recortes reales) |

**Por que este modelo:** YOLO detecta cajas e ignora que el tablero es una rejilla
conocida de 64 casillas. Aqui se rectifica el tablero, se recortan las 64 casillas
y se clasifica cada una. El split se hace **por tablero, nunca por casilla**: las
64 casillas de una misma foto caen siempre en el mismo split, si no hay fuga.

Antes de ejecutar: **Entorno de ejecucion > Cambiar tipo de entorno > T4 GPU**."""),
    md("## 1. Verificar la GPU"),
    code("""#@title 1. Verificar GPU y versiones
!nvidia-smi

import torch
print("PyTorch:", torch.__version__)
print("CUDA de PyTorch:", torch.version.cuda)
print("GPU disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("SIN GPU: ve a Entorno de ejecucion > Cambiar tipo de entorno > T4 GPU")"""),
    md("""## 2. Dependencias

`torch`, `torchvision` y `Pillow` ya vienen en Colab. La celda solo instala si
falta algo, y **nunca** reinstala `torch`: cambiarlo rompe la GPU de la sesion."""),
    code("""#@title 2. Comprobar dependencias sin tocar torch
import importlib.util
import subprocess
import sys

for paquete, modulo in (("torchvision", "torchvision"), ("pillow", "PIL")):
    if importlib.util.find_spec(modulo) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", paquete], check=True)

import torch
import torchvision
print("torch", torch.__version__, "| torchvision", torchvision.__version__)"""),
    md("""## 3. Traer el paquete de datos

El zip trae:

```
casillas/<clase>/<archivo>.jpg     152,192 recortes de 100x100
split/train.txt, valid.txt, test.txt   una ruta relativa por linea
labels/clases.txt                    las 13 clases en orden
labels/*.csv                         los CSV de referencia del etiquetado
```

La lista de cada split es la que manda: no hay `ImageFolder` ni carpetas por
split. Se descomprime en `/content` (disco local), no en Drive: 152 mil archivos
pequeños leidos desde Drive son lentisimos.

**Sube `clasificador_v2.zip` a `Mi unidad/ajedrez_v2/`** y deja `USAR_DRIVE = True`."""),
    code("""#@title 3. Descomprimir clasificador_v2.zip y contar por split
import shutil
import zipfile
from pathlib import Path

USAR_DRIVE = True
ZIP_EN_DRIVE = Path("/content/drive/MyDrive/ajedrez_v2/clasificador_v2.zip")
DESTINO = Path("/content/ajedrez_v2/clasificador")
ESPERADO = {"train": 115584, "valid": 19072, "test": 17536}

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
        f"No encuentro {origen}. Sube clasificador_v2.zip a Mi unidad/ajedrez_v2/ "
        "o pon USAR_DRIVE = False."
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
print("Salida:", PROYECTO)"""),
    md("""## 4. Dataset, transformaciones y pesos de clase

Un `Dataset` minimo que lee la lista de rutas del `.txt` y saca la clase del
**nombre de la carpeta** (`casillas/<clase>/archivo.jpg`). Nada de indices
guardados: la lista de split es la unica fuente de verdad.

- `Resize(128, 128)` + `Normalize` con las medias de ImageNet (el modelo es
  preentrenado, espera esa normalizacion).
- Solo en train, un `ColorJitter` suave: las dos sesiones de luz se parecen mas.
- **Pesos de clase** = frecuencia inversa sobre el split de train. `vacia` es el
  63.8% de las casillas; sin el peso el modelo aprende a decir "vacia" y listo."""),
    code("""#@title 4. Datasets, loaders y pesos de clase
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

RAIZ = DESTINO
TAMANO = 128
BATCH = 128
EPOCAS = 12
LR = 1e-3
SEMILLA = 42
NUM_WORKERS = 4

torch.manual_seed(SEMILLA)

TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize((TAMANO, TAMANO)),
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
    \"\"\"Casillas de un split, leidas desde su .txt (rutas relativas al zip).\"\"\"

    def __init__(self, raiz: Path, txt: Path, clases: list[str], transform):
        self.raiz = raiz
        self.clases = clases
        self.transform = transform
        self.rutas = [l.strip() for l in txt.read_text(encoding="utf-8").splitlines()
                      if l.strip()]

    def __len__(self) -> int:
        return len(self.rutas)

    def __getitem__(self, indice: int):
        relativa = self.rutas[indice]
        # casillas/<clase>/<archivo>.jpg
        clase = relativa.split("/")[1]
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
pesos = pesos / pesos.mean()  # media 1: no mueve la escala del learning rate

print(f"{'clase':12s} {'train':>8s} {'%':>7s} {'peso':>6s}")
for clase, peso in zip(CLASES, pesos.tolist()):
    n = conteo_train[clase]
    print(f"{clase:12s} {n:8d} {100 * n / totales:6.2f}% {peso:6.2f}")
print(f"\\nvacia = {100 * conteo_train['vacia'] / totales:.2f}% del train | "
      f"dama_b/vacia = {conteo_train['dama_b'] / conteo_train['vacia']:.3f}")
print("items:", {k: len(v) for k, v in conjuntos.items()})"""),
    md("""## 5. Modelo, optimizador y funcion de perdida

`mobilenet_v3_small` preentrenado en ImageNet: 2.5 M de parametros con la cabeza
de 1000 clases, ~1.5 M al cambiarla por la de 13. Entra de sobra en una T4 y en el
ESP32/edge del pipeline. La cabeza es una `Linear` de 13 salidas que reemplaza la
de 1000 clases.

`AdamW` con `lr = 1e-3` + cosine, y `CrossEntropyLoss(weight=...)` con los pesos
calculados arriba."""),
    code("""#@title 5. Crear el modelo
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"


def crear_modelo(n_clases: int) -> nn.Module:
    try:
        red = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except (AttributeError, TypeError):  # torchvision viejo
        red = models.mobilenet_v3_small(pretrained=True)
    for parametro in red.features.parameters():
        parametro.requires_grad = True
    entrada = red.classifier[3].in_features
    red.classifier[3] = nn.Linear(entrada, n_clases)
    return red


modelo = crear_modelo(len(CLASES)).to(DISPOSITIVO)
optimizador = AdamW(modelo.parameters(), lr=LR, weight_decay=1e-4)
planificador = CosineAnnealingLR(optimizador, T_max=EPOCAS)
criterio = nn.CrossEntropyLoss(weight=pesos.to(DISPOSITIVO))

n_parametros = sum(p.numel() for p in modelo.parameters())
print("Dispositivo:", DISPOSITIVO)
print(f"Parametros: {n_parametros / 1e6:.2f} M")
print("Salida:", modelo.classifier[3].out_features, "clases")"""),
    md("""## 6. Entrenar (12 epocas)

Se guarda en Drive cada vez que `valid` mejora. Si Colab se desconecta, el
`best_clasificador.pt` sigue en tu unidad."""),
    code("""#@title 6. Entrenar
import copy
import time
from pathlib import Path

import torch


def evaluar(red, cargador) -> tuple[float, float]:
    red.eval()
    perdida_total = 0.0
    aciertos = 0
    vistos = 0
    with torch.no_grad():
        for imagenes, etiquetas in cargador:
            imagenes = imagenes.to(DISPOSITIVO, non_blocking=True)
            etiquetas = etiquetas.to(DISPOSITIVO, non_blocking=True)
            salida = red(imagenes)
            perdida_total += criterio(salida, etiquetas).item() * len(etiquetas)
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
        salida = modelo(imagenes)
        perdida = criterio(salida, etiquetas)
        perdida.backward()
        optimizador.step()
        perdida_total += perdida.item() * len(etiquetas)
        aciertos += (salida.argmax(1) == etiquetas).sum().item()
        vistos += len(etiquetas)
    planificador.step()

    perdida_train = perdida_total / max(vistos, 1)
    acc_train = aciertos / max(vistos, 1)
    perdida_val, acc_val = evaluar(modelo, cargadores["valid"])
    segundos = time.time() - arranque
    historial.append((epoca, perdida_train, acc_train, perdida_val, acc_val))
    print(f"epoca {epoca:2d}/{EPOCAS}  loss={perdida_train:.4f} train={acc_train:.3f} "
          f"valid={acc_val:.3f}  ({segundos:.0f}s)")

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
print("Pesos:", SALIDA / "best_clasificador.pt")"""),
    md("""## 7. Matriz de confusion por clase en `valid` y `test`

El accuracy global esconde el problema real: `vacia` es el 62% de las casillas. Si
el modelo solo acierta `vacia` y las piezas comunes, el numero se ve bien y el
pipeline falla. Aqui se ve clase por clase."""),
    code("""#@title 7. Matriz de confusion y reporte por clase
import numpy as np
import torch


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
    print()"""),
    md("""## 8. Guardar el modelo final en Drive

`best_clasificador.pt` guarda el `state_dict`, la lista de clases y el tamano de
entrada. Para usarlo despues hay que reconstruir `mobilenet_v3_small` y cargar el
`state_dict`: el archivo no trae la arquitectura, solo los pesos."""),
    code("""#@title 8. Verificar que el modelo se puede recargar y listar Drive
import shutil

import torch

SALIDA = PROYECTO / "entrega"
SALIDA.mkdir(parents=True, exist_ok=True)

shutil.copy2(PROYECTO / "best_clasificador.pt", SALIDA / "best_clasificador.pt")
(SALIDA / "clases.txt").write_text("\\n".join(CLASES) + "\\n", encoding="utf-8")

comprobacion = crear_modelo(len(CLASES)).to(DISPOSITIVO)
comprobacion.load_state_dict(torch.load(SALIDA / "best_clasificador.pt",
                                        map_location=DISPOSITIVO)["modelo"])
comprobacion.eval()
print("Recarga OK:", SALIDA / "best_clasificador.pt")
for p in sorted(SALIDA.iterdir()):
    print(f"  {p.name:28s} {p.stat().st_size / 1024:8.1f} KB")"""),
    md("""## Como entra esto en el pipeline v2

Una vez entrenado, el pipeline (`src/`) hace, en este orden:

1. **Detector YOLO** (`best.pt` del otro notebook): encuentra las cajas de las
   fichas en el frame completo.
2. **Rectificacion del tablero**: con las 4 esquinas se aplica la homografia y el
   tablero queda en 800x800 con casillas de 100x100.
3. **Clasificador por casilla** (`best_clasificador.pt`): cada una de las 64
   casillas se recorta, se redimensiona a 128x128 y se clasifica en una de las 13
   clases. `vacia` no es un fracaso: es informacion.
4. **Filtro de movimientos legales con `python-chess`**: se construye el FEN con
   las 64 predicciones y se descartan las posiciones que un tablero legal no
   permite (dos reyes por bando, peones en la primera fila, etc.). Ese post-filtro
   es el que convierte "64 predicciones sueltas" en una posicion de ajedrez.

La comparacion de T7 (**YOLO solo vs clasificador solo vs combinado**) se hace
sobre el **mismo test real**: los recortes reales de `test` son exactamente los
mismos que usa el detector, porque el reparto se hizo por imagen fuente."""),
    md("""## Si algo falla

| Sintoma | Causa | Solucion |
|---|---|---|
| `SIN GPU` | Entorno sin GPU | Entorno de ejecucion > Cambiar tipo de entorno > T4 GPU |
| `No encuentro ...clasificador_v2.zip` | Ruta de Drive distinta | Revisa `ZIP_EN_DRIVE` o pon `USAR_DRIVE = False` |
| Descomprimir tarda mucho | 152 mil archivos pequeños | Es normal la primera vez: 3-5 minutos |
| `CUDA out of memory` | `BATCH` alto | Baja `BATCH` a 64 |
| `accuracy` alta pero el pipeline falla | El modelo dice `vacia` a todo | Mira la matriz por clase: si `recall` de piezas es bajo, sube epocas o el peso de clase |
| DataLoader lento | `/content` lleno o `NUM_WORKERS` bajo | Sube `NUM_WORKERS` a 8 y vuelve a descomprimir |
| Se perdio el entrenamiento | Colab cerro la sesion | El `best_clasificador.pt` esta en Drive: no hay nada que rehacer |"""),
]


def main() -> int:
    for nombre, celdas in (
        ("ajedrez_v2_detector.ipynb", DETECTOR),
        ("ajedrez_v2_clasificador.ipynb", CLASIFICADOR),
    ):
        ruta = AQUI / nombre
        ruta.write_text(json.dumps(notebook(celdas, nombre), indent=1, ensure_ascii=False),
                        encoding="utf-8")
        code_cells = sum(1 for c in celdas if c["cell_type"] == "code")
        print(f"{ruta}  celdas={len(celdas)} ({code_cells} codigo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
