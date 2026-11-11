"""Diagnostico puntual: que dijo el clasificador en las casillas conflictivas."""
import sys
from pathlib import Path

RAIZ = Path(r"F:\TAVO\Projects\ajedrez-v2")
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))
import cv2  # noqa: E402
from pipeline_fen import cargar_clasificador, clasificar_recorte  # noqa: E402

modelo, clases, _ = cargar_clasificador(RAIZ / "modelos" / "best_clasificador.pt")
base = RAIZ / "resultados" / "prueba_fen_declared"
CASILLAS = ("f7", "g7", "h7", "b7", "e4", "d4")
for idx in (1, 2, 3):
    img = cv2.imread(str(base / f"declared_{idx}_cenital.jpg"))
    occ = clasificar_recorte(modelo, clases, img)
    partes = []
    for casilla in CASILLAS:
        d = occ[casilla]
        partes.append(f"{casilla}={d['elegida']}({d['confianza']:.2f})")
    print(f"captura {idx}:  " + "  ".join(partes))
