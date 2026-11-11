"""Aplica las correcciones del usuario sobre revision.csv.

Las reglas (dictadas por el usuario el 2026-11-06, revisando los montajes):
  * c6 = caballo_n en FILES donde se movio el caballo negro.
  * c5 = alfil_n donde el usuario marco 'c5 es caballo_n-debe ser alfil_n'.
  * Casos especificos (a6/f6 peones en 044318, b5=alfil_b en capture (19),
    d6=caballo_n en capture (22), e6 en 28/29/30/32/33/36/37,
    f6 en 38/39/40/41/42/43/44/47/48).
  * capture (31) y capture (34): SE DESCARTAN (la orientacion del tablero
    quedo volteada y el mapeo de casillas no es confiable).

  Todo lo no nombrado queda con la prediccion del clasificador ('uso
  directo', por el 'lo demas esta ok' del usuario).
"""
from __future__ import annotations

import csv
from pathlib import Path

REVISION = Path(r"F:\TAVO\Projects\ajedrez-v2") / "dataset" / "revision_humana"
CSV = REVISION / "revision.csv"

C6 = ["20261105_043309", "20261105_043313", "20261105_043321",
      "20261105_043326", "20261105_043343", "20261105_043347",
      "20261105_043355", "20261105_043359", "20261105_044310",
      "20261105_044321", "20261105_044340", "20261105_044343",
      "20261105_044346", "20261105_044359", "20261105_044404",
      "cap1", "cap2", "cap3", "cap4", "cap5", "cap6", "cap7", "cap8", "cap9",
      "cap10", "cap12", "capture (12)", "capture (13)", "capture (14)",
      "capture (15)", "capture (16)", "capture (2)", "capture (17)",
      "capture (19)", "capture (2)", "capture (20)", "capture (21)",
      "capture (23)", "capture (24)", "capture (25)", "capture (26)",
      "capture (27)"]

C5 = ["20261105_043309", "cap1", "cap2", "cap3", "cap4", "cap5", "cap6",
      "cap7", "cap8", "cap9", "cap10", "cap12", "capture (10)",
      "capture (13)", "capture (14)", "capture (15)", "capture (16)"]

ESPECIFICAS = {
    "20261105_044318": {"a6": "peon_n", "c6": "caballo_n", "f6": "peon_n"},
    "capture (19)": {"c6": "caballo_n", "b5": "alfil_b"},
    "capture (22)": {"d6": "caballo_n"},
    "capture (28)": {"e6": "caballo_n"},
    "capture (29)": {"e6": "caballo_n"},
    "capture (30)": {"e6": "caballo_n"},
    "capture (32)": {"e6": "caballo_n"},
    "capture (33)": {"e6": "caballo_n"},
    "capture (36)": {"e6": "caballo_n"},
    "capture (37)": {"e6": "caballo_n"},
    "capture (38)": {"f6": "caballo_n"},
    "capture (39)": {"f6": "caballo_n"},
    "capture (40)": {"f6": "caballo_n"},
    "capture (41)": {"f6": "caballo_n"},
    "capture (42)": {"f6": "caballo_n"},
    "capture (43)": {"f6": "caballo_n"},
    "capture (44)": {"f6": "caballo_n"},
    "capture (47)": {"f6": "caballo_n"},
    "capture (48)": {"f6": "caballo_n"},
}

DESCARTAR = {"capture (31)", "capture (34)"}


def main() -> int:
    filas = list(csv.DictReader(CSV.open(encoding="utf-8-sig", newline="")))
    cambios = 0
    descartadas = 0
    for fila in filas:
        archivo_base = fila["archivo"].split("/")[-1].rsplit(".", 1)[0]
        casilla = fila["casilla"]
        if archivo_base in DESCARTAR:
            fila["verdict"] = "DESCARTADA"
            descartadas += 1
            continue
        if archivo_base in C6 and casilla == "c6":
            fila["verdict"] = "caballo_n"
            cambios += 1
        elif archivo_base in C5 and casilla == "c5":
            fila["verdict"] = "alfil_n"
            cambios += 1
        elif archivo_base in ESPECIFICAS and casilla in ESPECIFICAS[archivo_base]:
            fila["verdict"] = ESPECIFICAS[archivo_base][casilla]
            cambios += 1

    with CSV.open("w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=filas[0].keys())
        escritor.writeheader()
        escritor.writerows(filas)

    archivos_desc = sorted({f["archivo"] for f in filas
                            if "DESCARTADA" in f.get("verdict", "")})
    print(f"verdicts corregidos por el usuario: {cambios}")
    print(f"filas marcadas DESCARTADA: {descartadas} "
          f"({len(archivos_desc)} archivos)")
    for a in archivos_desc:
        print("  -", a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
