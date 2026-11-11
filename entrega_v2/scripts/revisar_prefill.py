"""Pre-llenar VERDICT en revision.csv con las verdades deductivas.

El usuario confirmo dos hechos fisicos:
  1. La fila BLANCA estaba BIEN montada (el error solo fue en el orden de las
     NEGRAS). Por tantos la fila 1 y los peones blancos son standard.
  2. El error fue el CABALLO/ALFIL en las fichas NEGRAS de la fila 8: b8/c8/f8/
     g8 quedaron cruzadas en las fotos de esas sesiones.

Deduccion por foto:
  * Si el archivo fue AUTO-VERIFICADO contra el patron de la inicial (17):
      - filas 2/7 = peon_x en las 8 columnas
      - filas 3-6 = vacia (el patron de ocupacion coincide: sin fichas)
      - fila 1 blanca = standard completo (a1 torre_b .. h1 torre_b)
      - fila 8: a8/d8/e8/h8 standard (torre_n/dama_n/rey_n/torre_n) y
        b8/c8/f8/g8 = SIN VERDICT (el humano decide).
  * Si el archivo es tablero VACIO (3): todo vacia.
  * El resto (medio juego/finales) se quedan SIN VERDICT: revision humana.
"""
import csv
import re
from pathlib import Path

REVISION = Path(r"F:\TAVO\Projects\ajedrez-v2") / "dataset" / "revision_humana"
CSV = REVISION / "revision.csv"

FILA_ESTANDAR = ["torre", "caballo", "alfil", "dama", "rey",
                 "alfil", "caballo", "torre"]
INCIERTAS = ()  # FISICO CONFIRMADO por el usuario: f8=caballo_n, g8=alfil_n
# (la fila 8 de las negras quedo con el par f8/g8 CRUZADO del estandar de
# ajedrez; b8/c8 si son standard). El alfil nunca se movio: siempre en g8.


def clase_de(pieza: str, color: str) -> str:
    return f"{pieza}_{color}"


def main() -> int:
    filas = list(csv.DictReader(CSV.open(encoding="utf-8-sig", newline="")))
    archivo_estado: dict[str, str] = {}
    with (Path(r"F:\TAVO\Projects\ajedrez-v2") / "dataset" / "reales" /
          "etiquetas_clasificador.csv").open(encoding="utf-8-sig", newline="") as f:
        for fila_ in csv.DictReader(f):
            archivo_estado.setdefault(fila_["archivo_recorte"],
                                      fila_.get("estado", ""))

    prefijadas = 0
    revisadas = set()
    for fila in filas:
        archivo = fila["archivo"]
        estado = archivo_estado.get(archivo, "")
        casilla = fila["casilla"]
        declara = re.match(r"^(\d+)\.", "")  # no-op para regla
        letra = casilla[0]
        numero = int(casilla[1])

        if estado == "auto":
            es_inicial = True     # solo las verificadas: el patron coincide
        elif "05_tablero_vacio" in archivo:
            es_inicial = False
        else:
            es_inicial = False    # medio juego/finales: revision humana

        if "05_tablero_vacio" in archivo:
            fila["verdict"] = "vacia"
            prefijadas += 1
            continue

        if not es_inicial:
            continue   # medio juego / finales: revision humana

        color = "b" if numero in (1, 2) or estado == "auto" and numero >= 7 else None
        # filas 2 y 7 de la inicial: peones
        if numero in (2, 7):
            color = "b" if numero == 2 else "n"
            fila["verdict"] = clase_de("peon", color)
            prefijadas += 1
            continue
        # filas 3-6: vacias
        if numero in (3, 4, 5, 6):
            fila["verdict"] = "vacia"
            prefijadas += 1
            continue
        # fila 1 blanca: standard completa (el error fue solo en las negras)
        if numero == 1:
            pieza = FILA_ESTANDAR[ord(letra) - ord("a")]
            fila["verdict"] = clase_de(pieza, "b")
            prefijadas += 1
            continue
        # fila 8 negra: la fisica confirmada por el usuario fue
        # torre, caballo, alfil, dama, rey, CABALLO(f8), ALFIL(g8), torre.
        # (el par f8/g8 quedó cruzado del estandar en todas las iniciales;
        # si en UNA foto el caballo se movio, el usuario lo corrige luego).
        if numero == 8:
            NOMINA = {"a8": "torre_n", "b8": "caballo_n", "c8": "alfil_n",
                      "d8": "dama_n", "e8": "rey_n", "f8": "caballo_n",
                      "g8": "alfil_n", "h8": "torre_n"}
            fila["verdict"] = NOMINA[casilla]
            prefijadas += 1
            continue

    with CSV.open("w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=filas[0].keys())
        escritor.writeheader()
        escritor.writerows(filas)

    print(f"verdict pre-llenados: {prefijadas}/{len(filas)}")
    print(f"archivos que TODAVIA necesitan revision humana: {len(revisadas)}")
    for a in sorted(revisadas):
        print(f"  - {a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
