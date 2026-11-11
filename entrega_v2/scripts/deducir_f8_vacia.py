"""Deduce f8 vacia en las fuentes 'auto' con ficha movida al centro.

Hecho fisico: en ALGUNAS fotos el caballo negro se movio desde f8 a una
casilla del centro (el usuario lo corrigio por casilla: c6/d6/e6/f6 etc).
El autoetiquetado inicial poblo f8=caballo_n SIEMPRE: para los archivos auto
con ficha real en filas 3..6, f8 debe ser VACIA (si no, la posicion termina
con 33 piezas y python-chess la rechaza). Tambien repara revision.csv para
que quede consistente si se vuelve a aplicar.
"""
import csv
from pathlib import Path

REVISION_DIR = Path(r"F:\TAVO\Projects\ajedrez-v2") / "dataset" / "revision_humana"
ETIQUETAS = (Path(r"F:\TAVO\Projects\ajedrez-v2") / "dataset" / "reales" /
             "etiquetas_clasificador.csv")
CSV_REVISION = REVISION_DIR / "revision.csv"


def filas_centro(tabla: dict[str, str]) -> dict[str, str]:
    return {c: clase for c, clase in tabla.items()
            if c and len(c) == 2 and c[0] in "abcdefgh"
            and int(c[1]) in (3, 4, 5, 6)}


def main() -> int:
    filas = list(csv.DictReader(ETIQUETAS.open(encoding="utf-8-sig", newline="")))
    por_archivo: dict[str, dict[str, str]] = {}
    estados: dict[str, set[str]] = {}
    for fila in filas:
        if not fila["casilla"]:
            continue
        por_archivo.setdefault(fila["archivo_recorte"], {})[fila["casilla"]] = fila["clase"]
        estados.setdefault(fila["archivo_recorte"], set()).add(fila["estado"])

    ajustars: list[str] = []
    for archivo, tabla in por_archivo.items():
        if not archivo.startswith("01_inicial/"):
            continue
        ocupadas = {c: clase for c, clase in tabla.items()
                    if clase and clase != "vacia"}
        if len(ocupadas) < 33:
            continue
        hay_centro = any(clase not in ("", "vacia")
                         for clase in filas_centro(tabla).values())
        if not hay_centro:
            continue
        ajustars.append(archivo)
        # f8 es el casillero del caballo que se movio: queda vacia. Si ya
        # tenia 'vacia' no cambia; si tenia ficha, se vacia y el archivo
        # pierde una cabeza fantasma que no existe fisicamente.
        for fila in filas:
            if fila["archivo_recorte"] == archivo and fila["casilla"] == "f8":
                fila["clase"] = "vacia"
                fila["estado"] = "revisada"
                fila["motivo"] = "el caballo se movio: f8 queda vacia"

    with (ETIQUETAS).open("w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=filas[0].keys())
        escritor.writeheader()
        escritor.writerows(filas)

    # Consistencia en revision.csv (por si se vuelve a aplicar)
    if CSV_REVISION.exists():
        fila2 = list(csv.DictReader(CSV_REVISION.open(encoding="utf-8-sig", newline="")))
        for fila in fila2:
            if (fila["archivo"] not in ajustars
                    or fila["casilla"] != "f8"):
                continue
            fila["verdict"] = "vacia"
        with CSV_REVISION.open("w", encoding="utf-8-sig", newline="") as f:
            escritor = csv.DictWriter(f, fieldnames=fila2[0].keys())
            escritor.writeheader()
            escritor.writerows(fila2)

    print("archivos auto con el caballo movido al centro (f8->vacia):")
    for a in ajustars:
        print("  -", a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
