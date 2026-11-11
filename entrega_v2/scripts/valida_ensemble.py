"""Ensemble de los DOS pesos: promedio de softmax.
Compara viejo/nuevo/ensemble en los dos dominios (compuesto + minidataset).
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torchvision

RAIZ = Path(r"F:\TAVO\Projects\ajedrez-v2")
sys.path.insert(0, str(RAIZ / "src"))


def cargar(peso: Path):
    import torch.nn as nn

    ck = torch.load(peso, map_location="cpu", weights_only=False)
    clases = ck["clases"]
    net = torchvision.models.mobilenet_v3_small(weights=None)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, len(clases))
    net.load_state_dict(ck["modelo"])
    net.eval()
    return net, clases


def main() -> int:
    import torchvision.transforms as T

    pre = T.Compose([T.ToTensor(), T.Resize((128, 128), antialias=True),
                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    viejo, clases = cargar(RAIZ / "modelos" / "best_clasificador.pt")
    nuevo, _ = cargar(RAIZ / "modelos" / "best_clasificador_luz.pt")
    indice = {c: i for i, c in enumerate(clases)}

    def predecir_tres(ruta: Path) -> tuple[int, int, int]:
        img = cv2.imread(str(ruta))
        if img is None:
            return (-1, -1, -1)
        tensor = pre(cv2.cvtColor(np.ascontiguousarray(img),
                                  cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            p1 = viejo(tensor)[0].numpy()
            p2 = nuevo(tensor)[0].numpy()
        return (int(p1.argmax()), int(p2.argmax()), int((p1 + p2).argmax()))

    random.seed(11)
    test_txt = (RAIZ / "dataset" / "colab_bundle" / "clasificador_split" /
                "test.txt").read_text(encoding="utf-8").splitlines()
    compuesto = random.sample([l.strip() for l in test_txt if l.strip()], 600)
    minidataset = [ruta for clase in clases
                   for ruta in sorted((RAIZ / "dataset" / "minidataset" /
                                       "casillas" / clase).glob("*.jpg"))]

    for etiqueta, grupo, mapeo in (
        ("compuesto", compuesto, lambda v: v.split("/")[-2]),
        ("minidataset", minidataset, lambda v: v.parent.name),
    ):
        datos = {}
        modelos = {"viejo": 0, "nuevo": 1, "ensemble": 2}
        for menos, net_ in (("viejo", viejo), ("nuevo", nuevo),
                            ("ensemble", (viejo, nuevo))):
            pass
        scores = {"viejo": Counter(), "nuevo": Counter(), "ensemble": Counter()}
        totals = Counter()
        for ruta in grupo:
            verd = mapeo(ruta)
            resultado = predecir_tres(ruta)
            if resultado[0] < 0:
                continue
            totals[verd] += 1
            for etiqueta_2, indice_resultado in zip(("viejo", "nuevo", "ensemble"),
                                                    resultado):
                if indice_resultado == indice[verd]:
                    scores[etiqueta_2][verd] += 1
        print(f"== {etiqueta} ==")
        for etiqueta_2 in ("viejo", "nuevo", "ensemble"):
            aciertos = sum(scores[etiqueta_2][c] for c in ("__total__",))
            np_total = sum(totals.values())
            print(f"  {etiqueta_2}: {sum(scores[etiqueta_2].values())}/{np_total} "
                  f"({100.0 * sum(scores[etiqueta_2].values()) / max(np_total, 1):.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
