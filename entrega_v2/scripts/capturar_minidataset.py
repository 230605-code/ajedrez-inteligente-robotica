"""Captura el MINI-DATASET para el fine-tune del clasificador.

Etiqueta automatica: en la POSICION INICIAL las 64 casillas son las del
estandar (blancas a1..h1 + peones 2; negras a8..h8 + peones 7, con el orden
FISICO CONFIRMADO por el usuario) y el resto vacias.

El uso Tobago:
    python scripts/capturar_minidataset.py --ip 10.251.192.200 --n 16 --segondo 8
    * durante la secuencia, CAMBIA LA LUZ entre la captura 8 y la 9 (el script
      hace una pausa larga justo ahi).
    * NO muevas fichas durante estas capturas.

Cada captura produce:
    dataset/minidataset/frames/frame_NN.jpg          (crudo)
    dataset/minidataset/cenital/frame_NN.jpg         (800x800 rectificado)
    dataset/minidataset/casillas/<clase>/fNN_<casilla>.jpg   (128x128 cada casilla)
    dataset/minidataset/etiquetas.csv                (archivo_foto, casilla, clase)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from esquinas_aruco import detectar_robusto  # noqa: E402

DESTINO_CENITAL = np.array([[0, 0], [799, 0], [799, 799], [0, 799]],
                           dtype="float32")

# La posicion INICIAL estandar de ajedrez (ya corregida: b8/c8/f8/g8 fisica).
BLANCAS = {"a1": "torre_b", "b1": "caballo_b", "c1": "alfil_b", "d1": "dama_b",
           "e1": "rey_b", "f1": "alfil_b", "g1": "caballo_b", "h1": "torre_b"}
NEGRAS = {"a8": "torre_n", "b8": "caballo_n", "c8": "alfil_n", "d8": "dama_n",
          "e8": "rey_n", "f8": "alfil_n", "g8": "caballo_n", "h8": "torre_n"}
VERDAD_INICIAL = dict(BLANCAS)
VERDAD_INICIAL.update(NEGRAS)
VERDAD_INICIAL.update({f"{c}2": "peon_b" for c in "abcdefgh"})
VERDAD_INICIAL.update({f"{c}7": "peon_n" for c in "abcdefgh"})
for fila in ("3", "4", "5", "6"):
    VERDAD_INICIAL.update({f"{c}{fila}": "vacia" for c in "abcdefgh"})


def capturar(url: str) -> np.ndarray | None:
    import requests

    try:
        respuesta = requests.get(url, timeout=15)
        respuesta.raise_for_status()
        return cv2.imdecode(np.frombuffer(respuesta.content, np.uint8),
                            cv2.IMREAD_COLOR)
    except requests.RequestException as exc:
        print(f"  ERROR {type(exc).__name__}: {str(exc)[:80]}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="10.251.192.200")
    parser.add_argument("--url")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--segundo", type=float, default=8.0)
    parser.add_argument("--corte_luz", type=int, default=8,
                        help="en esa captura se pausa largo: cambia la luz ahi")
    parser.add_argument("--mover", default="",
                        help="casillas desplazadas en el frame, tipo "
                             "'b1=vacia,f3=caballo_b,g8=vacia,c6=caballo_n' "
                             "(sobreescribe la verdad estandar)")
    args = parser.parse_args()

    url = args.url or f"http://{args.ip}/capture"
    base = RAIZ / "dataset" / "minidataset"
    verdad = dict(VERDAD_INICIAL)
    overrides: dict[str, str] = {}
    if args.mover:
        for pareja in args.mover.split(","):
            casilla, clase = pareja.split("=")
            overrides[casilla.strip()] = clase.strip()
        verdad.update(overrides)
        print(f"movimientos aplicados a la verdad: {overrides}")
    for carpeta in ("frames", "cenital"):
        (base / carpeta).mkdir(parents=True, exist_ok=True)
    (base / "casillas").mkdir(parents=True, exist_ok=True)

    etiquetas: list[dict[str, str]] = []
    contador_clases: dict[str, int] = {}

    for indice in range(1, args.n + 1):
        nombre = f"frame_{indice:02d}"
        print(f"\n--- captura {indice}/{args.n} ({nombre})",
              flush=True)
        if indice == args.corte_luz:
            print("  >>> CAMBIA LA LUZ AHORA (enciende/apaga de ser posible) <<<",
                  flush=True)
            for restante in range(int(12), 0, -1):
                print(f"    {restante}s...", end="\r", flush=True)
                cv2.waitKey(1000) if False else __import__("time").sleep(1)
            print()

        frame = capturar(url)
        if frame is None:
            print("  sin frame: reintenta en 6s")
            __import__("time").sleep(6)
            frame = capturar(url)
            if frame is None:
                continue
        cv2.imwrite(str(base / "frames" / f"{nombre}.jpg"), frame)

        esquinas, metodo, detalle = detectar_robusto(frame)
        if esquinas is None:
            print(f"  sin esquinas ({detalle or metodo}): frame guardado sin casillas")
            continue
        print(f"  esquinas {metodo}: {len(detalle)} chars")

        matriz = cv2.getPerspectiveTransform(
            np.asarray(esquinas, dtype="float32"), DESTINO_CENITAL)
        cenital = cv2.warpPerspective(frame, matriz, (800, 800),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))
        cv2.imwrite(str(base / "cenital" / f"{nombre}.jpg"), cenital)

        import torchvision.transforms as T

        preparar = T.Compose([T.ToPILImage(), T.Resize((128, 128), antialias=True)])
        for fila in range(8):
            for columna in range(8):
                casilla = f"{chr(ord('a') + columna)}{8 - fila}"
                clase = verdad[casilla]
                recorte = cenital[fila * 100:(fila + 1) * 100,
                                  columna * 100:(columna + 1) * 100]
                recorte = cv2.resize(recorte, (128, 128),
                                     interpolation=cv2.INTER_AREA)
                (base / "casillas" / clase).mkdir(parents=True, exist_ok=True)
                destino = (base / "casillas" / clase
                           / f"f{indice:02d}_{casilla}.jpg")
                cv2.imwrite(str(destino), recorte)
                etiquetas.append({"archivo_foto": nombre,
                                  "casilla": casilla, "clase": clase,
                                  "archivo": str(destino.relative_to(base))})
                contador_clases[clase] = contador_clases.get(clase, 0) + 1
        if indice < args.n:
            __import__("time").sleep(max(2.0, args.segundo))

    import csv

    with (base / "etiquetas.csv").open("w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=["archivo_foto", "casilla",
                                                "clase", "archivo"])
        escritor.writeheader()
        escritor.writerows(etiquetas)
    print(f"\nmini-dataset listo: {len(etiquetas)} casillas en {base}")
    print(f"por clase: {dict(sorted(contador_clases.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
