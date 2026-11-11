"""Verificacion LOCAL del best.pt entrenado en Colab (detector v2).

Reproduce el val de test (56 imagenes reales) fuera de Colab:
  * orden de clases contra classes.txt
  * metricas generales y por clase sobre val/test del bundle
Escribe la comparacion contra la linea base medida (modelo_sintetico v1:
mAP50 0.040 sobre este mismo test).
"""
import re
import shutil
import sys
import tempfile
from pathlib import Path

RAIZ = Path(r"F:\TAVO\Projects\ajedrez-v2")
BUNDLE = RAIZ / "dataset" / "colab_bundle" / "detector"
MODELO = RAIZ / "modelos" / "best.pt"
sys.path.insert(0, str(RAIZ / "scripts"))
from comun import cargar_clases  # noqa: E402

clases = cargar_clases()

from ultralytics import YOLO  # noqa: E402

modelo = YOLO(str(MODELO))
nombres_modelo = [modelo.names[i] for i in range(len(modelo.names))]
coincide = nombres_modelo == clases
print("clases del modelo:")
for indice, nombre in enumerate(nombres_modelo):
    marca = "OK" if indice < len(clases) and nombre == clases[indice] else "DESDORDEN!"
    print(f"  {indice:2d} {nombre:12s} {marca}")
if not coincide:
    raise SystemExit("El orden de clases NO coincide con classes.txt: no evaluar")

# Val con el yaml del bundle reescrito a path absoluto LOCAL y RUTA SHARED
# (el datasets_dir global del usuario puede ser cualquiera: no depende de el)
temporal = Path(tempfile.mkdtemp(prefix="val_v2_", dir=str(BUNDLE)))
try:
    (BUNDLE / "data.yaml").write_text(
        re.sub(r"^path: .*$", f"path: {BUNDLE.as_posix()}",
               (BUNDLE / "data.yaml").read_text(encoding="utf-8"),
               flags=re.M),
        encoding="utf-8",
    )
    shutil.copyfile(BUNDLE / "data.yaml", temporal / "data.yaml")
    yaml_texto = (temporal / "data.yaml").read_text(encoding="utf-8")
    print(f"yaml temporal: {temporal / 'data.yaml'}")
    print(yaml_texto.splitlines()[:6])

    for split in ("test", "val"):
        metricas = modelo.val(
            data=str(temporal / "data.yaml"), split=split,
            imgsz=640, batch=8, verbose=False,
            project=str(RAIZ / "resultados" / "verificacion_v2"),
            name=f"local_{split}", exist_ok=True,
        )
        print(f"\n[{split}] fold: mAP50 {metricas.box.map50:.4f}  "
              f"mAP50-95 {metricas.box.map:.4f}  P {metricas.box.mp:.4f}  "
              f"R {metricas.box.mr:.4f}")
        if split == "test":
            print("  por clase (P/R/mAP50):")
            for indice, clase in enumerate(metricas.names.values()):
                componente = metricas.box.class_result(indice)
                p, r, ap50, _ = componente
                print(f"  {clase:12s} P={p:.3f} R={r:.3f} mAP50={ap50:.3f}")
finally:
    shutil.rmtree(temporal, ignore_errors=True)
