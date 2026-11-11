"""Aplica las correcciones del usuario ( posicion avanzada 2026-11-07 ) al
fen.txt de las pruebas y deja la version corregida lista para la tesis.

Correcciones dictadas por el usuario:
  a7 = torre_n   (el modelo dijo peon_n 50%)
  g4 = vacia     (el modelo dijo caballo_n 90%: fanasma)
  h4 = peon      (el modelo dijo peon_b; el color no es distintivo aqui)
  a3 = caballo   (el modelo dijo caballo_b; es del lado NEGRO fisico)
  h2 = vacia     (coincidio)
"""
from __future__ import annotations

from pathlib import Path

RAIZ = Path(r"F:\TAVO\Projects\ajedrez-v2")
SALIDA = RAIZ / "assets" / "previews" / "revista_viva"

FEN_BASE = "2kr1b1r/p1p3p1/b1nq1p1p/1p1pp3/2B1P1nP/NQ1P1NP1/PP1P1P2/R1B2RK1 w - - 0 1"


def aplicar(fen: str, correcciones: dict[str, str]) -> str:
    partes, turno_extra = fen.split(" ")[0], fen.split(" ", 1)[1]
    filas = partes.split("/")
    tabla: dict[str, str] = {}
    for fila_i, fila in enumerate(filas):
        columna = 0
        for letra in fila:
            if letra.isdigit():
                for vacia in range(int(letra)):
                    casilla = f"{chr(ord('a') + columna)}{8 - fila_i}"
                    tabla[casilla] = ""
                    columna += 1
            else:
                casilla = f"{chr(ord('a') + columna)}{8 - fila_i}"
                tabla[casilla] = letra
                columna += 1
    for casilla, texto in correcciones.items():
        tabla[casilla] = "" if texto == "-" else texto
    nueva_filas = []
    for fila_i in range(8):
        numerica = ""
        vacias = 0
        for c in range(8):
            letra = tabla.get(f"{chr(ord('a') + c)}{8 - fila_i}", "")
            if letra in ("", "vacia"):
                vacias += 1
                continue
            if vacias:
                numerica += str(vacias)
                vacias = 0
            numerica += letra
        if vacias:
            numerica += str(vacias)
        nueva_filas.append(numerica)
    return "/".join(nueva_filas) + " " + turno_extra


def main() -> int:
    corregida = aplicar(FEN_BASE, {
        "a7": "r",   # torre_n (usuario), en vecso peon
        "g4": "-",   # vacia (fanasma del modelo eliminstada)
        "h4": "p",   # peon negro fisico (disco claro)
        "a3": "n",   # caballo negro fisico (disco claro)
        "h2": "",
    })
    print("FEN sin filtros   : ", FEN_BASE)
    print("FEN CORREGIDA     : ", corregida)
    for indice in (1, 2):
        ruta = SALIDA / f"prueba_{indice:02d}" / "fen.txt"
        if ruta.exists():
            texto = ruta.read_text(encoding="utf-8")
            texto += f"\nFEN CORREGIDA POR EL USUARIO:\n  {corregida}\n"
            ruta.write_text(texto, encoding="utf-8")
    print("(escrito en fen.txt de prueba_01 y prueba_02)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
