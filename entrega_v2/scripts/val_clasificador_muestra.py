"""Evaluar el clasificador v2 (best_clasificador.pt) contra su test real.

El checkpoint guarda 'modelo' (state_dict) y 'clases'. Re-mobile el
MobileNetV3-Small, carga los pesos y evalua sobre una muestra del split test
(listado dentro del zip, rutas relativas a la raiz del proyecto). CPU: la
muestra queda en ~2000 casillas (~5 min).
"""
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

RAIZ = Path(r"F:\TAVO\Projects\ajedrez-v2")
RUTA = Path(r"F:\TAVO\Projects\ajedrez-v2\modelos\best_clasificador.pt")
MUESTRA = 2000

ck = torch.load(RUTA, map_location="cpu", weights_only=False)
clases: list[str] = ck["clases"]
if isinstance(ck["modelo"], dict):  # es state_dict: reconstruyo arquitectura
    net = models.mobilenet_v3_small(weights=None)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, len(clases))
    net.load_state_dict(ck["modelo"])
else:
    net = ck["modelo"]
net.eval()

rutas: list[str] = []
blob = RAIZ / "dataset" / "colab_bundle" / "clasificador_split" / "test.txt"
rutas = [l.strip() for l in blob.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"clases ({len(clases)}): {clases}")
print(f"test: {len(rutas)} casillas; muestreando {MUESTRA} con semilla fija")
random.seed(42)
muestra = random.sample(rutas, min(MUESTRA, len(rutas)))

transformacion = None
try:
    from torchvision import transforms as T

    transformacion = T.Compose([
        T.ToTensor(),
        T.Resize((128, 128), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"sin torchvision transforms: {exc}")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

aciertos = 0
total = 0
por_clase: Counter = Counter()
por_clase_total: Counter = Counter()
confusion: Counter = Counter()

imagen_preparada = {}
for ruta_virtual in muestra:
    clase_correcta = ruta_virtual.split("/")[-2]
    clas_id = clases.index(clase_correcta)
    if ruta_virtual not in imagen_preparada:
        # rutinas virtuales del split: casillas/<clase>/archivo.jpg
        # compuestos en  dataset/reales_compuesto/casillas/clasificador/<clase>
        # reales-aug  en  dataset/reales/clasificador/casillas/<clase>
        ruta_local = (RAIZ / "dataset" / "reales_compuesto"
                      / ruta_virtual.replace("casillas/", "casillas/clasificador/", 1))
        if not ruta_local.exists():
            ruta_local = (RAIZ / "dataset" / "reales"
                          / "clasificador" / ruta_virtual)
        img = cv2.imread(str(ruta_local)) if ruta_local.exists() else None
        if img is None:
            continue
        imagen_preparada[ruta_virtual] = transformacion(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tensor = imagen_preparada[ruta_virtual].unsqueeze(0)
    with torch.no_grad():
        salida = net(tensor)[0]
    prob = torch.softmax(salida, 0)
    predicho = int(prob.argmax().item())
    por_clase_total[clase_correcta] += 1
    if predicho == clas_id:
        aciertos += 1
        por_clase[clase_correcta] += 1
    else:
        confusion[(clase_correcta, clases[predicho])] += 1
    total += 1

print(f"\naciertos sobre la muestra: {aciertos}/{total} "
      f"({100.0 * aciertos / max(total, 1):.1f}%)")
print("clase -> recall muestra (siempre incluye las bien):")
for clase in clases:
    n = por_clase_total[clase]
    if n == 0:
        print(f"  {clase:10s} (sin muestras)")
        continue
    print(f"  {clase:10s} {100.0 * por_clase[clase] / n:5.1f}%  ({por_clase[clase]}/{n})")
print("confusiones mas frecuentes (esperado -> predicho):")
for (verdadera, predicha), n in confusion.most_common(8):
    print(f"  {verdadera:10s} -> {predicha:10s} {n}")
