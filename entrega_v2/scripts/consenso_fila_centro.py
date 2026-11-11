"""Consenso dual sobre las casillas 'fantasma' del centro (filas 3-6).

Problema medido: en 73 recortes de 01_inicial, el clasificador predice
caballo/alfil en casillas del medio SIEMPRE con 1.0 de confianza para las
MISMAS casillas (c4=alfil_b, f3=caballo_b...) y en otras es la ficha real que
el usuario movio. Imposible separarlo con un solo modelo.

Regla del consenso:
  de cada casilla candidata, si el DETECTOR YOLO no ve NINGUNA ficha con su
  centro dentro de esa casilla -> VERDICT = vacia (era espejismo).
  si el DETECTOR si ve ficha -> se deja la prediccion del clasificador para
  revision humana (montaje de sobrevivientes).

Guarda un montaje por archivo con las casillas candidatas agrandadas 3x para
que el usuario confirme. Escribe los verdict de los espejismos directamente
en revision.csv.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2

RAIZ = Path(r"F:\TAVO\Projects\ajedrez-v2")
REVISION = RAIZ / "dataset" / "revision_humana"
MONTE = REVISION / "consenso_fila_centro"
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from pipeline_fen import realinear_por_fichas  # noqa: E402


def celda_de(cx: float, cy: float) -> tuple[int, int] | None:
    """(fila 0..7, columna 0..7) de un centro; None si esta fuera."""
    if not (0.0 <= cx < 800.0 and 0.0 <= cy < 800.0):
        return None
    fila, columna = int(cy // 100), int(cx // 100)
    if 3 <= fila <= 6:
        return fila, columna
    return None


def main() -> int:
    from ultralytics import YOLO

    detector = YOLO(str(RAIZ / "modelos" / "best.pt"))

    def detectar(imagen):
        res = detector.predict(imagen, conf=0.25, agnostic_nms=True,
                               verbose=False)[0]
        return [
            (res.names[int(b.cls.item())],
             float((b.xyxy[0][0] + b.xyxy[0][2]) / 2.0),
             float((b.xyxy[0][1] + b.xyxy[0][3]) / 2.0),
             float(b.conf.item()))
            for b in res.boxes
        ]

    filas = list(csv.DictReader((REVISION / "revision.csv")
                                .open(encoding="utf-8-sig", newline="")))
    filas.sort(key=lambda f: (f["archivo"], f["casilla"]))

    # Agrupa las casillas candidatas por archivo (filas 3..6 de 01_inicial)
    candidatos: dict[str, list[csv.Dict]] = defaultdict(list)
    for fila in filas:
        numero = int(fila["casilla"][1])
        if ("01_inicial" not in fila["archivo"] or
                not 3 <= numero <= 6 or
                not fila["clase_predecida"].startswith(("caballo", "alfil"))):
            continue
        candidatos[fila["archivo"]].append(fila)

    MONTE.mkdir(parents=True, exist_ok=True)
    (MONTE / "01_inicial").mkdir(parents=True, exist_ok=True)

    por_verdict: dict[str, list[str]] = defaultdict(list)
    montajes = []
    total_vaciar = total_sobrevive = 0
    for archivo, lista in sorted(candidatos.items()):
        ruta = RAIZ / "dataset" / "reales" / "recortes" / "800x800" / archivo
        imagen = cv2.imread(str(ruta))
        if imagen is None:
            continue
        alineada, motivo = realinear_por_fichas(imagen, detectar)
        marble = alineada if alineada is not None else imagen
        fichas = detectar(marble)
        mapa = {}
        for clase, cx, cy, conf in fichas:
            celda = celda_de(cx, cy)
            if celda is not None:
                mapa[f"celda{celda[0]}{celda[1]}"] = (clase, conf)

        tira = marble.copy()
        for fila in lista:
            fila_i = 8 - int(fila["casilla"][1])
            col_i = ord(fila["casilla"][0]) - ord("a")
            celda = (fila_i, col_i)
            llave = f"celda{fila_i}{col_i}"
            hay = mapa.get(llave)
            if hay is None:
                fila["verdict"] = "vacia"     # espectro: nada con centro ahi
                por_verdict["vaciado"].append(fila["casilla"])
                cv2.putText(tira, f"{fila['casilla']}: vacia", (col_i * 100 + 4,
                            fila_i * 100 + 90), 4, 0.42, (20, 220, 30), 1)
            else:
                por_verdict["sobrevive"].append(f"{fila['casilla']}={hay[0]} {hay[1]:.2f}")
                cv2.putText(tira, f"{fila['casilla']}: {hay[0]}",
                            (col_i * 100 + 4, fila_i * 100 + 90),
                            4, 0.42, (0, 200, 255), 1)

        bloques = []
        for f_ in lista:
            fila_i = 8 - int(f_["casilla"][1])
            col_i = ord(f_["casilla"][0]) - ord("a")
            recorte = marble[fila_i * 100:(fila_i + 1) * 100,
                             col_i * 100:(col_i + 1) * 100]
            recorte = cv2.resize(recorte, None, fx=3, fy=3,
                                 interpolation=cv2.INTER_CUBIC)
            cv2.putText(recorte, f_["casilla"], (6, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            vered = f_.get("verdict") or "?"
            cv2.putText(recorte, vered, (6, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (30, 255, 40) if vered == "vacia" else (0, 220, 255), 2)
            bloques.append(recorte)
        if bloques:
            fila_mag = cv2.hconcat(bloques[:4])
            montajes.append(fila_mag)   # solo la primera corrida por archivo
        nombre = archivo.split("/")[-1]
        cv2.imwrite(str(MONTE / "01_inicial" /
                        ("realineado_" + nombre.replace(" ", "_"))), tira)

    # Escribimos de vuelta los verdict en revision.csv.
    destino = REVISION / "revision.csv"
    with destino.open("w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=filas[0].keys())
        escritor.writeheader()
        escritor.writerows(filas)

    if montajes:
        maxima = max(m.shape[1] for m in montajes)
        sal = cv2.vconcat([m if m.shape[1] == maxima
                           else cv2.copyMakeBorder(m, 0, 0, 0, maxima - m.shape[1],
                                                   cv2.BORDER_CONSTANT)
                           for m in montajes])
        cv2.imwrite(str(REVISION / "consenso_sobrevivientes.jpg"), sal)

    print("espejismos descartados (verdict=vacia):",
          len(por_verdict["vaciado"]))
    print("sobrevivientes para el humano:",
          len(por_verdict["sobrevive"]))
    for dato in sorted(set(por_verdict["sobrevive"]))[:20]:
        print("   ", dato)
    print(f"no realineados/montaje: {REVISION / 'consenso_fila_centro'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
