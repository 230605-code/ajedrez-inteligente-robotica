"""Restaura el estado 'auto' de las 20 fuentes fisicamente verificadas.

Las 17 de posicion inicial + 3 tableros vacios. Sin este paso, el
pre-llenado deductivo no puede reconocerlas (porque una corrida anterior
del --aplicar las reescribio todas como 'revisada' y perdimos la pista de
cual fuente era la posicion fisica estandar).
"""
import csv
from pathlib import Path

ETIQUETAS = (Path(r"F:\TAVO\Projects\ajedrez-v2") / "dataset" / "reales" /
             "etiquetas_clasificador.csv")

AUTO = [
    "01_inicial/20261104_231551.jpg", "01_inicial/20261104_231605.jpg",
    "01_inicial/20261104_231618.jpg", "01_inicial/20261104_231631.jpg",
    "01_inicial/20261104_231654.jpg", "01_inicial/20261104_231700.jpg",
    "01_inicial/20261104_231705.jpg", "01_inicial/20261104_231710.jpg",
    "01_inicial/20261104_231715.jpg", "01_inicial/20261104_231726.jpg",
    "01_inicial/20261104_231757.jpg", "01_inicial/capture (11).jpg",
    "01_inicial/capture (2).jpg", "01_inicial/capture (3).jpg",
    "01_inicial/capture (6).jpg", "01_inicial/capture (7).jpg",
    "01_inicial/capture (8).jpg", "01_inicial/capture (9).jpg",
    "05_tablero_vacio/20261105_054643.jpg",
    "05_tablero_vacio/20261105_054651.jpg",
    "05_tablero_vacio/20261105_054705.jpg",
]


def main() -> int:
    filas = list(csv.DictReader(ETIQUETAS.open(encoding="utf-8-sig", newline="")))
    restauradas = 0
    for fila in filas:
        if fila["archivo_recorte"] in AUTO:
            if fila["estado"] != "auto":
                fila["estado"] = "auto"
                fila["motivo"] = "estado restaurado (fuente verificada)"
                restauradas += 1
    with ETIQUETAS.open("w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=filas[0].keys())
        escritor.writeheader()
        escritor.writerows(filas)
    print(f"filas restauradas a estado 'auto': {restauradas} "
          f"(pendientes: 20 fuentes fisicas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
