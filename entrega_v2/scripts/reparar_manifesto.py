"""Repara el manifesto de una carpeta: borra las filas cuyos jpg ya no estan."""
import csv
import sys
from pathlib import Path

carpeta = RAIZ = Path(r"F:\TAVO\Projects\ajedrez-v2") / "dataset" / "crudo" / (
    sys.argv[1] if len(sys.argv) > 1 else "03_medio_juego")

ruta = carpeta / "manifesto.csv"
filas = list(csv.DictReader(ruta.open(encoding="utf-8-sig", newline="")))
validas = []
borradas = []
for fila in filas:
    archivo = (fila.get("archivo") or "").strip()
    if archivo and (carpeta / archivo).exists():
        validas.append(fila)
    elif archivo:
        borradas.append((archivo, fila.get("posicion", "?")))

with ruta.open("w", encoding="utf-8-sig", newline="") as f:
    escritor = csv.DictWriter(f, fieldnames=["archivo", "posicion", "fen", "fuente"])
    escritor.writeheader()
    escritor.writerows(validas)

print(f"manifesto de {carpeta.name}: {len(validas)} filas validas")
if borradas:
    print(f"borradas del manifesto ({len(borradas)}):")
    for archivo, posicion in borradas:
        print(f"  - {archivo} (posicion {posicion})")
else:
    print("ninguna habia sido borrada del disco")
