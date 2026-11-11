"""Auditoria de etiquetas YOLO antes de la T5 (entrenamiento).

Verifica lo que hace que el entrenamiento falle en silencio:
  1. Estructura: 5 campos por linea, id de clase dentro de classes.txt,
     coordenadas en [0, 1], ancho/alto no degenerados.
  2. Coincidencia con el manifiesto: cajas por imagen == n_fichas declarado.
  3. Orientacion y orden: el id 0..11 debe corresponder al orden de
     classes.txt para ambos datasets.
  4. Muestra visual: dibuja cajas sobre 3 imagenes al azar por dataset
     con sus clases para inspeccion humana.
"""
from __future__ import annotations

import csv
import random
import sys
from collections import Counter
from pathlib import Path

import cv2

RAIZ = Path(r"F:\TAVO\Projects\ajedrez-v2")
sys.path.insert(0, str(RAIZ / "scripts"))
from comun import cargar_clases  # noqa: E402

PREVIEWS = RAIZ / "assets" / "previews" / "auditoria"
CONJUNTOS = {
    "sintetico": (RAIZ / "dataset" / "sintetico" / "images",
                  RAIZ / "dataset" / "sintetico" / "labels",
                  RAIZ / "dataset" / "sintetico" / "manifesto.csv",
                  "n_fichas"),
    "compuesto": (RAIZ / "dataset" / "reales_compuesto" / "images",
                  RAIZ / "dataset" / "reales_compuesto" / "labels",
                  RAIZ / "dataset" / "reales_compuesto" / "manifiesto.csv",
                  "n_fichas"),
}


def auditar(nombre: str, imagenes: Path, labels: Path,
            manifiesto: Path, campo_fichas: str, clases: list[str]) -> Counter:
    avisos = Counter()
    declarado: dict[str, int] = {}
    if manifiesto.exists():
        with manifiesto.open(encoding="utf-8-sig", newline="") as f:
            for fila in csv.DictReader(f):
                declarado[fila.get("imagen", fila.get("archivo", ""))] = int(
                    float(fila.get(campo_fichas, "0") or 0)
                )
    solo_imagenes = {p.stem for p in imagenes.glob("*.jpg")}
    solo_labels = {p.stem for p in labels.glob("*.txt")}
    avisos["sin_label"] += len(solo_imagenes - solo_labels)
    avisos["sin_imagen"] += len(solo_labels - solo_imagenes)

    ids_vistos: Counter = Counter()
    cajas_totales = 0
    fuera_rango = 0
    distinto_declarado = 0
    random.seed(7)
    muestra = random.sample(sorted(solo_labels), min(3, len(solo_labels)))

    for label in labels.glob("*.txt"):
        cajas_imagen = 0
        texto = label.read_text(encoding="utf-8")
        for linea in texto.splitlines():
            partes = linea.split()
            if not partes:
                avisos["linea_vacia"] += 1
                continue
            if len(partes) != 5:
                avisos["campos"] += 1
                continue
            try:
                clas_id, x, y, w, h = (float(v) for v in partes)
            except ValueError:
                avisos["numerico"] += 1
                continue
            if int(clas_id) < 0 or int(clas_id) >= len(clases):
                avisos["id_fuera"] += 1
                continue
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
                    and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                fuera_rango += 1
            cajas_imagen += 1
            ids_vistos[int(clas_id)] += 1
        cajas_totales += cajas_imagen
        esperado = declarado.get(label.stem + ".jpg")
        if esperado is not None and esperado != cajas_imagen:
            distinto_declarado += 1

    print(f"\n=== {nombre} ===")
    print(f"  imagenes: {len(solo_imagenes)}  labels: {len(solo_labels)}")
    print(f"  cajas totales: {cajas_totales}")
    print(f"  coords fuera de [0,1]: {fuera_rango}")
    print(f"  desbalance con n_fichas: {distinto_declarado}")
    for k, v in avisos.items():
        print(f"  AVISO {k}: {v}")
    faltan_clases = [c for i, c in enumerate(clases) if i not in ids_vistos]
    print(f"  clases presentes: {len(ids_vistos)}/{len(clases)}",
          f"faltan: {faltan_clases}" if faltan_clases else "")
    print(f"  cajas por clase: {dict(sorted(ids_vistos.items()))}")

    # muestra visual con clases impresas
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    for stem in muestra:
        imagen = cv2.imread(str(imagenes / f"{stem}.jpg"))
        if imagen is None:
            continue
        for linea in (labels / f"{stem}.txt").read_text(encoding="utf-8").splitlines():
            partes = linea.split()
            if len(partes) != 5:
                continue
            cid, cx, cy, w, h = (float(v) for v in partes)
            point_x, point_y = int((cx - w / 2) * imagen.shape[1]), int((cy - h / 2) * imagen.shape[0])
            point_x2, point_y2 = int((cx + w / 2) * imagen.shape[1]), int((cy + h / 2) * imagen.shape[0])
            etiqueta_txt = clases[int(cid)]
            cv2.rectangle(imagen, (point_x, point_y), (point_x2, point_y2), (40, 200, 40), 1)
            cv2.putText(imagen, etiqueta_txt, (point_x, max(10, point_y - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        ruta_salida = PREVIEWS / f"{nombre}_{stem}.jpg"
        cv2.imwrite(str(ruta_salida), imagen)
    print(f"  previews: {len(muestra)} en assets/previews/auditoria/")
    return avisos


def main() -> int:
    clases = cargar_clases()
    print("classes.txt:", clases)
    contador_total: Counter = Counter()
    for nombre, (img, lab, mf, campo) in CONJUNTOS.items():
        if not img.exists():
            print(f"\n=== {nombre}: NO EXISTE {img}")
            continue
        for k, v in auditar(nombre, img, lab, mf, campo, clases).items():
            contador_total[k] += v
    bloqueadores = [k for k in ("id_fuera", "sin_label", "sin_imagen",
                                "campos", "numerico") if contador_total[k]]
    print("\nVEREDICTO:", "OK para T5" if not bloqueadores
          else f"BLOQUEOS: {bloqueadores}")
    return 1 if bloqueadores else 0


if __name__ == "__main__":
    raise SystemExit(main())
