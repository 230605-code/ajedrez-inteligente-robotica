"""Probe directo: pasa el ARTE IMPRINT de cada ficha por el clasificador.

Si el clasificador etiqueta mal el CABALLO del arte impreso (que es la fuente
de verdad del objeto fisico), el error esta EN EL ENTRENAMIENTO (clases
cruzadas o sprite equivocado). Si lo etiqueta bien, el problema es del luz
de la camara en vivo, no de las etiquetas.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"F:\TAVO\Projects\ajedrez-v2\src")
sys.path.insert(0, r"F:\TAVO\Projects\ajedrez-v2\scripts")
from pipeline_fen import cargar_clasificador  # noqa: E402

import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402

tr = T.Compose([T.ToTensor(), T.Resize((128, 128), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])
net, clases, _ = cargar_clasificador(
    Path(r"F:\TAVO\Projects\ajedrez-v2\modelos\best_clasificador.pt"))

img = cv2.imread(r"F:\TAVO\Projects\ajedrez-v2\plantilla\fichas.png")
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
circulos = cv2.HoughCircles(gris, cv2.HOUGH_GRADIENT, dp=1.2, minDist=55,
                            param1=110, param2=35, minRadius=20, maxRadius=70)
puntos = sorted(((int(y), int(x), int(r)) for x, y, r in circulos[0]))
grupos: list[list[tuple[int, int, int]]] = [[puntos[0]]]
for punto in puntos[1:]:
    if punto[0] - grupos[-1][-1][0] > 40:
        grupos.append([punto])
    else:
        grupos[-1].append(punto)
# filas de fichas.png de arriba a abajo: [0] pawns blancas, [1] piezas
# blancas, [2] pawns negras, [3] piezas negras. El layout impreso es
# torre caballo alfil dama rey alfil caballo torre.
# (la plantilla imprime cada 'juego' en 2 filas de 8: peones + piezas)
orden_impreso = ["torre", "caballo", "alfil", "dama", "rey",
                 "alfil", "caballo", "torre"]

# La fila de piezas de CADA color es la que tiene 8 discos curvas.
filas_8 = [g for g in grupos if len(g) == 8]
print(f"filas de 8: {len(filas_8)}")
# En fichas.png filas de 8 son: [0] peones blancas (y 8), [1] piezas blancas,
# [2] peones negras, [3] piezas negras -> peones = primera de cada par.
fila_pieza_b = None
fila_pieza_n = None
for indice, grupo in enumerate(filas_8):
    y_medio = sum(p[0] for p in grupo) / len(grupo)
    if indice == 0:
        pass
for indice, grupo in enumerate(filas_8):
    y_medio = sum(p[0] for p in grupo) / len(grupo)
    arriba = "arriba" if y_medio < 200 else "abajo"
    print(f"  fila {indice}: {len(grupo)} disos, y medio {y_medio:.0f} ({arriba})")

fila_pieza_b = filas_8[1] if len(filas_8) >= 2 else None
fila_pieza_n = filas_8[3] if len(filas_8) >= 4 else None

for nombre_fila, fila, color in (("BLANCAS", fila_pieza_b, "b"),
                                 ("NEGRAS", fila_pieza_n, "n")):
    if fila is None:
        continue
    fila_ordenada = sorted(fila, key=lambda t: t[1])
    print(f"\n--- {nombre_fila} (arte impreso -> lo que DICE el clasificador)")
    for col, pieza_fisica in enumerate(orden_impreso):
        x, y, r = fila_ordenada[col]
        r_ = r + 4
        disco = img[max(y - r_, 0):y + r_, max(x - r_, 0):x + r_]
        grande = cv2.resize(disco, None, fx=2, fy=2,
                            interpolation=cv2.INTER_CUBIC)
        entrada = tr(cv2.cvtColor(np.ascontiguousarray(grande),
                                  cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(net(entrada)[0], 0).numpy()
        orden = np.argsort(probs)[::-1]
        esp = clases[int(orden[0])]
        bien = (esp == f"{pieza_fisica}_{color}")
        print(f"  {col+1}. {pieza_fisica:8s}{color} -> {esp:11s} "
              f"{probs[orden[0]]:.2f}  {'OK' if bien else '<< MAL'}")
