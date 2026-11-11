"""Repara y COMPLETA los manifestos de crudo (01..05):
   (a) borra las filas cuyos .jpg ya no existen
   (b) registra los .jpg que el usuario agrego y no tenian entrada
       (posicion desde el nombre pNN; si no tiene pNN斌 toma la posicion
       maxima declarada de esa carpeta: era la "posicion actual")
   (c) la FEN sale del mapa declarado de cada carpeta:
       01 = estandar fisico, 05 = todo vacia, 02/03/04 = las FEN del
       manifesto ya existente (las declaradas por el usuario).

Se usa por carpeta:  python reparar_crudo.py 01_inicial_intacta
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

RAIZ = Path(r"F:\TAVO\Projects\ajedrez-v2")
CRUDO = RAIZ / "dataset" / "crudo"

BLANCAS = {"a1": "torre_b", "b1": "caballo_b", "c1": "alfil_b", "d1": "dama_b",
           "e1": "rey_b", "f1": "alfil_b", "g1": "caballo_b", "h1": "torre_b"}
NEGRAS = {"a8": "torre_n", "b8": "caballo_n", "c8": "alfil_n", "d8": "dama_n",
          "e8": "rey_n", "f8": "alfil_n", "g8": "caballo_n", "h8": "torre_n"}
ESTANDAR = {**BLANCAS, **NEGRAS,
            **{f"{c}2": "peon_b" for c in "abcdefgh"},
            **{f"{c}7": "peon_n" for c in "abcdefgh"},
            **{f"{c}{f}": "vacia" for c in "abcdefgh" for f in (3, 4, 5, 6)}}


def main() -> int:
    carpetas = sys.argv[1:] or [d.name for d in sorted(CRUDO.iterdir())
                                if d.is_dir()]
    for nombre in carpetas:
        carpeta = CRUDO / nombre
        if not carpeta.is_dir():
            continue
        ruta = carpeta / "manifesto.csv"
        mapa_fen: dict[int, str] = {}
        filas = []
        if ruta.exists():
            for fila_csv in csv.DictReader(ruta.open(encoding="utf-8-sig",
                                                      newline="")):
                archivo = (fila_csv.get("archivo") or "").strip()
                if not archivo or not (carpeta / archivo).exists():
                    continue  # (a) fila de un archivo borrado
                try:
                    id_pos = int(fila_csv.get("posicion") or 0)
                except ValueError:
                    id_pos = 0
                fen = (fila_csv.get("fen") or "").strip()
                if fen and fen != "(SIN DECLARAR)":
                    mapa_fen[id_pos] = fen
                fuente = (fila_csv.get("fuente") or "").strip() or "auto"
                filas.append({"archivo": archivo, "posicion": str(id_pos),
                              "fen": fen, "fuente": fuente})
        if nombre == "01_inicial_intacta":
            mapa_fen.setdefault(1, "inicial_estandar")
        if nombre == "05_tablero_vacio":
            mapa_fen[999] = "todo_vacia"

        # (b) archivos en el disco sin registro: son fotos del usuario de la
        #     "posicion actual" de esa carpeta.
        registrados = {f["archivo"] for f in filas}
        nuevas = []
        for ruta_img in carpeta.glob("*.jpg"):
            if ruta_img.name in registrados:
                continue
            coincidencia = re.search(r"p(\d{1,2})_", ruta_img.name)
            if coincidencia:
                id_pos = int(coincidencia.group(1))
            else:
                id_pos = max(mapa_fen.keys(), default=0) if mapa_fen else 0
            fen = mapa_fen.get(id_pos, "(SIN DECLARAR)")
            filas.append({"archivo": ruta_img.name, "posicion": str(id_pos),
                          "fen": fen, "fuente": "usuario"})
            nuevas.append(ruta_img.name)

        filas.sort(key=lambda f: (int(f["posicion"]), f["archivo"]))
        with ruta.open("w", encoding="utf-8-sig", newline="") as f:
            escritor = csv.DictWriter(f, fieldnames=["archivo", "posicion",
                                                    "fen", "fuente"])
            escritor.writeheader()
            escritor.writerows(filas)

        print(f"\n[{nombre}] {len(filas)} fotos en el manifesto "
              f"(+{len(nuevas)} nuevas del disco)")
        for p in range(1, 100):
            n = sum(1 for f_ in filas if int(f_["posicion"]) == p)
            if n:
                print(f"    posicion {p}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
