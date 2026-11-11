"""Mini fine-tune del clasificador con el dataset vivo (fix del caballo).

Entrena SOLO unos epochs con LR bajo sobre best_clasificador.pt usando las
casillas del minidataset (las de la luz real de la ESP32) mezcladas con una
muestra de las casillas compuesto (para no olvidar lo demas). Guarda el peso
en `modelos/best_clasificador_luz.pt` y deja intacto el modelo en uso hasta
que la prueba en vivo lo valide.
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(r"F:\TAVO\Projects\ajedrez-v2")

EPOCAS = 8
LR = 1e-4
BATCH = 16
SEMILLA = 7
MUESTRA_ANTIGUAS = 400

MINI = RAIZ / "dataset" / "minidataset" / "casillas"
COMPU = RAIZ / "dataset" / "reales_compuesto" / "casillas" / "clasificador"
PESO_BASE = RAIZ / "modelos" / "best_clasificador.pt"
PESO_SALIDA = RAIZ / "modelos" / "best_clasificador_luz.pt"


def casillas(carpeta_clases: Path) -> list[Path]:
    return [archivo for clase in sorted(carpeta_clases.iterdir())
            if clase.is_dir()
            for archivo in sorted(clase.glob("*.jpg"))][: -1]


def cargar_imagen(ruta: Path):
    img = cv2.imread(str(ruta))
    if img is None:
        return None
    return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2RGB)


def preparar(ruta: Path, transformacion):
    img = cargar_imagen(ruta)
    if img is None:
        raise SystemExit(f"imagen ilegible: {ruta}")
    return transformacion(img)


def main() -> int:
    import time

    import torch
    import torchvision.transforms as T

    random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)

    transformacion = T.Compose([
        T.ToTensor(),
        T.Resize((128, 128), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # El checkpoint guarda: 'modelo' (state_dict), 'clases'.
    import torch.nn as nn
    import torchvision

    ckpt = torch.load(PESO_BASE, map_location="cpu", weights_only=False)
    clases: list[str] = ckpt["clases"]
    indice_de = {c: i for i, c in enumerate(clases)}
    net = torchvision.models.mobilenet_v3_small(weights=None)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, len(clases))
    net.load_state_dict(ckpt["modelo"])

    # 1. Dataset del mini: TODAS las casillas del minidataset (etiqueta
    #    automatica por la posicion inicial).
    mini: list[tuple[Path, int]] = []
    for clase in clases:
        carpeta = MINI / clase
        for archivo in sorted(carpeta.glob("*.jpg")):
            mini.append((archivo, indice_de[clase]))
    print(f"mini-dataset: {len(mini)} casillas")

    # 2. Muestra de las antiguas (anti-olvido), balanceada por clase.
    antiguas: list[tuple[Path, int]] = []
    for clase in clases:
        carpeta = COMPU / clase
        todas = sorted(carpeta.glob("*.jpg"))
        n = max(8, MUESTRA_ANTIGUAS // len(clases))
        random.shuffle(todas)
        for archivo in todas[: min(n, len(todas))]:
            antiguas.append((archivo, indice_de[clase]))
    print(f"muestra antigua: {len(antiguas)} casillas")

    mezcla = mini + antiguas
    random.shuffle(mezcla)
    print(f"total entrenamiento: {len(mezcla)}")

    # 3. Entrenamiento congelando el grueso: SOLO el clasificador y el
    #    ultimo bloque ajustan (evita romper el resto de features).
    for nombre, parametro in net.named_parameters():
        parametro.requires_grad = "classifier" in nombre or \
                                  "features.12" in nombre

    optimizador = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad], lr=LR)
    perdida = nn.CrossEntropyLoss()

    inicio = time.time()
    paso = max(1, len(mezcla) // 3)
    for epoca in range(1, EPOCAS + 1):
        random.shuffle(mezcla)
        total = aciertos = 0
        perdida_media = 0.0
        net.train()
        for inicio_lote in range(0, len(mezcla), BATCH):
            lote = mezcla[inicio_lote:inicio_lote + BATCH]
            imagenes = torch.stack([
                preparar(ruta, transformacion) for ruta, _ in lote])
            objetivo = torch.tensor([clase for _, clase in lote])
            optimizador.zero_grad()
            salida = net(imagenes)
            perdida_lote = perdida(salida, objetivo)
            perdida_lote.backward()
            optimizador.step()
            perdida_media += float(perdida_lote) * len(lote)
            predicho = salida.argmax(1)
            aciertos += int((predicho == objetivo).sum())
            total += len(lote)
        print(f"  epoca {epoca}/{EPOCAS}: perdida {perdida_media / max(total, 1):.3f} "
              f"  aciertos {aciertos}/{total} ({100.0 * aciertos / max(total, 1):.1f}%) "
              f"({time.time() - inicio:.0f}s)", flush=True)

    torch.save({"modelo": net.state_dict(), "clases": clases,
                "epoca": EPOCAS, "tamano": 128,
                "fuente": "mini fine-tune con minidataset de la camara"},
               PESO_SALIDA)
    print(f"\nmodelo salido: {PESO_SALIDA}")
    print("USALO EN LA PRUEBA VIVO; si acierta el caballo, lo promovemos a"
          " best_clasificador.pt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
