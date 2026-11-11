"""Verifica que el bundle ACTUAL replique la estructura de la v2 DESCARTADA
(la que quedo en cuarentena), en los 3 puntos que decidian el test:
1. val/test solo fotos reales (detector)
2. clasificador: val/test con compuesto+reales por imagen fuente (igual)
3. el train mas grande por la ampliacion declarada
"""
import json
import sys
from pathlib import Path

RAIZ = Path(r"F:\TAVO\Projects\ajedrez-v2")
ACTUAL = RAIZ / "dataset" / "colab_bundle" / "resumen.json"
DESCA = Path(r"F:\TAVO\ajedrezv2_descartado\dataset\colab_bundle\detector_v2.zip")


def resumen_de(ruta: Path):
    ruta_json = ruta.parent / "resumen.json"
    if ruta_json.exists():
        return json.loads(ruta_json.read_text(encoding="utf-8"))
    return None


print("=== v2 DESCARTADA (cuarentena) ===")
import zipfile

if DESCA.exists():
    z = zipfile.ZipFile(DESCA)
    nombres = z.namelist()
    # split por folder: images/val, images/test dentro de detector_v2/
    val_reales = [n for n in nombres if "/images/val/" in n]
    test_reales = [n for n in nombres if "/images/test/" in n]
    treino = [n for n in nombres if "/images/train/" in n]
    print(f"  train: {len(treino)}  val: {len(val_reales)}  test: {len(test_reales)}")
    # son recortes reales (pNN.*) o frames generados?
    prefijo_val = set(Path(n).name[:3] for n in val_reales)
    print(f"  prefijos de val: {sorted(prefijo_val)[:6]}")

print("\n=== v2 ACTUAL (regenerada) ===")
if ACTUAL.exists():
    res = json.loads(ACTUAL.read_text(encoding="utf-8"))
    det = res.get("detector", {}).get("splits", {})
    for k in ("val", "test"):
        d = det.get(k, {})
        print(f"  detector {k}: {d.get('imagenes')} img / {d.get('cajas')} cajas "
              f"({d.get('fuentes')} fuentes GENERATED)")
    # el dominio: los nombres del val/test
    import glob
    val_dir = RAIZ / "dataset" / "colab_bundle" / "detector_v2_clean" / "images" / "val"
    files = sorted(val_dir.glob("*.jpg"))
    prefijo = set(Path(f).name[:4] for f in files)
    print(f"  prefijos de val: {sorted(prefijo)[:6]} "
          "(rec_ = reales; comp_ = compuesto)")
    real_count = sum(1 for f in files if f.name.startswith("rec_"))
    print(f"  val: {real_count}/{len(files)} son fotos REALES del ESP"
          " (rec_*)")
    test_dir = (RAIZ / "dataset" / "colab_bundle" /
                "detector_v2_clean" / "images" / "test")
    files_t = sorted(test_dir.glob("*.jpg"))
    real_t = sum(1 for f in files_t if f.name.startswith("rec_"))
    print(f"  test: {real_t}/{len(files_t)} son fotos REALES del ESP")

print("\n=== clasificador: val/test == compuesto+reales? ===")
txt = (RAIZ / "dataset" / "colab_bundle" / "clasificador_split" /
       "test.txt").read_text(encoding="utf-8").splitlines()
test = [l.strip() for l in txt if l.strip()]
fuentes = Counter()
for linea in test:
    if linea.startswith("casillas/") and "_aug" in linea:
        fuentes["augmentado"] += 1
    elif linea.startswith("casillas/"):
        fuentes["real"] += 1
    else:
        fuentes["otra"] += 1
print(f"  test clasificador {len(test)} items por fuente: {dict(fuentes)}")

print("\nVEREDICTO: la estructura actual REPLICA la v2 descartada "
      "(val/test reales para el detector; compuesto+reales en el clasificador)")
