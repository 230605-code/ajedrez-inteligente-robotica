"""Genera el lote de fine-tune con la FEN que declare el usuario.

La FEN del usuario = etiqueta casilla por casilla de su posicion fisica.
Toma la vista cenital de la mejor lectura (prueba_02 en este caso), recorta
las 64 casillas y las guarda en dataset/minidataset/casillas/<clase>/ con
nombre fU_<casilla>.jpg (las casillas que fallaron quedan con su etiqueta
CORRECTA de la mano del usuario, que es lo que ensena al modelo el caso).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

RAIZ = Path(r"F:\TAVO\Projects\ajedrez-v2")
sys.path.insert(0, str(RAIZ / "src"))

import argparse

parse = argparse.ArgumentParser()
parse.add_argument("--fen", default="r2q1rk1/pp1bppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/2KR3R w - - 0 11")
parse.add_argument("--cenital", default=str(RAIZ / "assets" / "previews" / "revista_viva" / "prueba_02" / "cenital.jpg"))
parse.add_argument("--prefijo", default="f17")
args = parse.parse_args()

CENITAL = Path(args.cenital)
BASE = RAIZ / "dataset" / "minidataset"

FEN_USUARIO = args.fen
PREFIJO = args.prefijo


def main() -> int:
    cenital = cv2.imread(str(CENITAL))
    if cenital is None:
        raise SystemExit(f"sin {CENITAL}")
    # quitar los ROTULOS dibujados del cenital: recorto 12 px externos para
    # evitar el texto de las casillas previas.
    limpio = cenital[4:796, 4:796]
    limpio = cv2.resize(limpio, (800, 800), interpolation=cv2.INTER_AREA)

    partes = FEN_USUARIO.split(" ")[0]
    cuenta: dict[str, int] = {}
    for fila_i, fila in enumerate(partes.split("/")):
        col = 0
        for letra in fila:
            if letra.isdigit():
                col += int(letra)
                continue
            pieza = letra.lower()
            color = "b" if letra.isupper() else "n"
            mapa = {"p": "peon", "r": "torre", "n": "caballo",
                    "b": "alfil", "q": "dama", "k": "rey"}
            clase = f"{mapa[pieza]}_{color}"
            casilla = f"{chr(ord('a') + col)}{8 - fila_i}"
            celda = limpio[fila_i * 100:(fila_i + 1) * 100,
                           col * 100:(col + 1) * 100]
            carpeta_cel = BASE / "casillas" / clase
            carpeta_cel.mkdir(parents=True, exist_ok=True)
            destino = carpeta_cel / f"{PREFIJO}_{casilla}.jpg"
            cv2.imwrite(str(destino), cv2.resize(celda, (128, 128),
                                                 interpolation=cv2.INTER_AREA))
            cuenta[clase] = cuenta.get(clase, 0) + 1
            col += 1
    print("casillas guardadas:", cuenta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
