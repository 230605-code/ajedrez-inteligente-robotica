"""Empaqueta entrega_v2.zip para jugar la interfaz desde otra PC.

Incluye: src, scripts, colab, docs, plantilla, modelos, tools/stockfish,
classes.txt, requirements.txt, iniciar_interfaz.py/.bat y README.md.
Excluye: __pycache__, *.pyc, *.zip internos y .venv.
"""
import zipfile
from pathlib import Path

raiz = Path(r"F:\TAVO\Projects\ajedrez-v2")
incluir = ["src", "scripts", "colab", "docs", "plantilla", "modelos",
           "classes.txt", "requirements.txt", "tools/stockfish",
           "iniciar_interfaz.py", "iniciar_interfaz.bat", "README.md"]
salida = raiz / "dataset" / "colab_bundle" / "entrega_v2.zip"
salida.parent.mkdir(parents=True, exist_ok=True)
if salida.exists():
    salida.unlink()
archivos = 0
with zipfile.ZipFile(salida, "w", zipfile.ZIP_DEFLATED) as zf:
    for item in incluir:
        origen = raiz / item
        if origen.is_file():
            zf.write(origen, item)
            archivos += 1
        elif origen.is_dir():
            for f in sorted(origen.rglob("*")):
                if (f.is_file() and "__pycache__" not in str(f)
                        and not f.name.endswith(".pyc")
                        and not f.name.endswith(".zip")
                        and ".venv" not in str(f)):
                    zf.write(f, str(f.relative_to(raiz)))
                    archivos += 1
print("zip:", salida)
print("peso MB:", round(salida.stat().st_size / 1024 / 1024, 1))
print("archivos:", archivos)
