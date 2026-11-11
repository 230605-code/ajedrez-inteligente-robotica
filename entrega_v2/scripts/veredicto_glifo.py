"""Veredicto por GLIFO: compara b8/c8/f8/g8 de la vista cenital viva contra los
sprites del arte impreso de plantilla/fichas.png (caballo_n y alfil_n), con las
4 rotaciones. Es un test de plantilla: no usa ningun modelo, por eso es la
fuente de verdad cuando todos desconfiamos de las etiquetas.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(r"F:\TAVO\Projects\ajedrez-v2")
CENITAL = RAIZ / "assets" / "previews" / "v2_vivo" / "v2_1_cenital.jpg"
FICHAS = RAIZ / "plantilla" / "fichas.png"
SALIDA = RAIZ / "assets" / "previews" / "glifo_veredicto"

CASILLAS = ["b8", "c8", "f8", "g8"]


def sprites_negras():
    """Del arte impreso: fila de piezas negras (discos claros, simbolo oscuro).

    fichas.png: 4 filas x 8 discos. Filas 0-1 = blancas (discos oscuros), 2-3
    = negras (discos claros). Fila 2 = peones. Fila 3 = torre caballo alfil
    dama rey alfil caballo torre.
    """
    imagen = cv2.imread(str(FICHAS))
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    circulos = cv2.HoughCircles(
        gris, cv2.HOUGH_GRADIENT, dp=1.2, minDist=55,
        param1=110, param2=35, minRadius=20, maxRadius=70,
    )
    if circulos is None:
        raise SystemExit("no encontre discos en fichas.png")
    puntos = [(int(x), int(y), int(r)) for x, y, r in circulos[0]]
    ordenados_y = sorted(puntos, key=lambda t: (t[1], t[0]))
    # Agrupa por distancia vertical: una fila nueva si el disco baja > 35 px.
    grupos: list[list[tuple[int, int, int]]] = [[ordenados_y[0]]]
    for punto in ordenados_y[1:]:
        if punto[1] - grupos[-1][-1][1] > 35:
            grupos.append([punto])
        else:
            grupos[-1].append(punto)
    # La fila de PIEZAS negras = la ultima fila completa de 8 discos.
    filas_8 = [g for g in grupos if len(g) == 8]
    if not filas_8:
        raise SystemExit("no identifique la fila de piezas negras")
    ordenados = sorted(filas_8[-1], key=lambda t: t[0])
    ficha_caballo = ordenados[1]
    ficha_alfil = ordenados[2]
    nombres = {}
    for nombre, (cx_, cy, radio) in (
        ("caballo_n", ficha_caballo), ("alfil_n", ficha_alfil),
    ):
        r_ = radio + 4
        disco = imagen[max(cy - r_, 0):cy + r_, max(cx_ - r_, 0):cx_ + r_]
        disco = cv2.resize(disco, (96, 96), interpolation=cv2.INTER_AREA)
        nombres[nombre] = disco
    return nombres


def puntuar(celda: np.ndarray, sprite: np.ndarray) -> tuple[float, int]:
    """Coincidencia (0..1) de la celda con el sprite en 4 rotaciones."""
    celda_ = cv2.resize(celda, (96, 96), interpolation=cv2.INTER_AREA)
    gris_c = cv2.cvtColor(celda_, cv2.COLOR_BGR2GRAY).astype("float32")
    gris_s = cv2.cvtColor(sprite, cv2.COLOR_BGR2GRAY).astype("float32")
    mejor = (float("inf"), 0)
    for giro in range(4):
        s = np.rot90(gris_s, giro)
        error = float(np.mean((gris_c - s) ** 2))
        if error < mejor[0]:
            mejor = (error, giro)
    return 1.0 / (1.0 + mejor[0] / 2000.0), mejor[1]


def main() -> int:
    cenital = cv2.imread(str(CENITAL))
    if cenital is None:
        raise SystemExit(f"No encuentro {CENITAL}; corre prueba_en_vivo_v2 antes")
    sprites = sprites_negras()
    SALIDA.mkdir(parents=True, exist_ok=True)
    print("veredicto por glifo (plantilla fisica vs casilla en vivo):")
    cuerpo = []
    for casilla in CASILLAS:
        columna = ord(casilla[0]) - ord("a")
        fila = 8 - int(casilla[1])
        celda = cenital[fila * 100:(fila + 1) * 100,
                        columna * 100:(columna + 1) * 100]
        fila_img = []
        rotulos = [f"{casilla} real"]
        for nombre, sprite in sprites.items():
            marca, giro = puntuar(celda, sprite)
            fila_img.append(celda)
            rotulos.append(f"{nombre} {marca:.2f} giro{giro * 90}")
        cuerpo.append(np.hstack(fila_img))
        print("  " + "   ".join(rotulos[0]) + "  |  " +
              "  ".join(rotulos[1:]))
        cuerpo.append(np.full((18, cuerpo[-1].shape[1], 3), 255, np.uint8))
    cv2.imwrite(str(SALIDA / "comparacion_glifos.jpg"), cv2.vconcat(cuerpo[:1] +
                  cuerpo[1:2]))
    print(f"comparacion guardada en {SALIDA / 'comparacion_glifos.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
