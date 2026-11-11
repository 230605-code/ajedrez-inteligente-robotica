"""Genera variantes aumentadas de los recortes REALES autoetiquetados.

Entrada: ``dataset/reales/recortes/800x800`` + ``dataset/reales/etiquetas_clasificador.csv``
(lo que produce ``autoetiquetar_reales.py``). Solo se aumentan los recortes con
``estado = auto``: los demas no tienen posicion conocida y sus casillas no
tendrian etiqueta.

Para cada recorte se generan 6 variantes deterministas (misma semilla -> mismos
archivos siempre) con:

* brillo/contraste 0.75..1.35, gamma 0.7..1.4, saturacion 0.8..1.2;
* ruido gaussiano sigma 3..12 y desenfoque gaussiano sigma 0..1.5;
* rotacion pequena +-5 grados y un jitter de perspectiva de hasta el 2% del
  borde.

El jitter de perspectiva NO se aplica a ciegas: se deforma la imagen completa y
las 64 casillas se recortan con la MISMA homografia, de modo que cada casilla
sigue cayendo exactamente donde debe. Un jitter que desplazara las casillas
respecto de la rejilla seria basura etiquetada como buena, y aqui se evita por
construccion: la etiqueta viaja con la geometria.

Salidas
-------
``dataset/reales/augmentado/images/<carpeta>/<nombre>_augK.jpg``
     las 6 variantes de cada recorte.
``dataset/reales/augmentado/etiquetas_clasificador.csv``
     las mismas columnas que el CSV de autoetiquetado mas ``variante``.
``dataset/reales/clasificador/casillas/<clase>/<nombre>_augK_<casilla>.jpg``
     64 recortes de 100x100 por variante, en una carpeta por clase (13).
``assets/previews/augmentado_muestra_1..4.jpg``
     hojas de contacto para revision humana.

Uso
---
    python aumentar_reales.py
    python aumentar_reales.py --limite 4 --sin-previews
"""
from __future__ import annotations

import argparse
import csv
import time
import zlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
RECORTES = RAIZ / "dataset" / "reales" / "recortes" / "800x800"
ETIQUETAS = RAIZ / "dataset" / "reales" / "etiquetas_clasificador.csv"
SALIDA = RAIZ / "dataset" / "reales" / "augmentado"
CASILLAS = RAIZ / "dataset" / "reales" / "clasificador" / "casillas"
PREVIEWS = RAIZ / "assets" / "previews"

LADO = 800
CASILLA = LADO // 8
VARIANTES = 6
SEMILLA = 20261105

CLASES = [
    "peon_b", "torre_b", "caballo_b", "alfil_b", "dama_b", "rey_b",
    "peon_n", "torre_n", "caballo_n", "alfil_n", "dama_n", "rey_n",
    "vacia",
]

CAMPOS = [
    "archivo_recorte", "casilla", "clase", "fila", "estado", "motivo", "variante",
]


def casilla(fila: int, columna: int) -> str:
    return f"{chr(ord('a') + columna)}{8 - fila}"


def _semilla(nombre: str, indice: int) -> int:
    """Semilla estable: crc32 y no hash(), que cambia en cada proceso."""
    return (zlib.crc32(nombre.encode("utf-8")) + SEMILLA + 7919 * indice) % (2 ** 32)


def homografia_variante(rng: np.random.Generator) -> np.ndarray:
    """Rotacion pequena + jitter de perspectiva de hasta 2% del borde."""
    angulo = float(rng.uniform(-5.0, 5.0))
    centro = (LADO / 2.0, LADO / 2.0)
    rotacion = cv2.getRotationMatrix2D(centro, angulo, 1.0)

    limite = 0.02 * LADO
    origen = np.array(
        [[0, 0], [LADO - 1, 0], [LADO - 1, LADO - 1], [0, LADO - 1]],
        dtype="float32",
    )
    destino = origen + rng.uniform(-limite, limite, size=(4, 2)).astype("float32")
    # La rotacion se compone despues del jitter: se aplica al destino.
    jitter = cv2.getPerspectiveTransform(origen, destino.astype("float32"))
    return jitter @ np.vstack([rotacion, [0.0, 0.0, 1.0]])


def fotometria(imagen: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Brillo, contraste, gamma, saturacion, ruido y desenfoque."""
    trabajo = imagen.astype("float32")

    contraste = float(rng.uniform(0.75, 1.35))
    brillo = float(rng.uniform(-25.0, 25.0))
    trabajo = (trabajo - 128.0) * contraste + 128.0 + brillo

    gamma = float(rng.uniform(0.7, 1.4))
    trabajo = np.clip(trabajo, 0.0, 255.0) / 255.0
    trabajo = np.power(trabajo, gamma) * 255.0

    saturacion = float(rng.uniform(0.8, 1.2))
    if abs(saturacion - 1.0) > 1e-3:
        hsv = cv2.cvtColor(np.clip(trabajo, 0, 255).astype("uint8"), cv2.COLOR_BGR2HSV)
        hsv = hsv.astype("float32")
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturacion, 0, 255)
        trabajo = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR).astype("float32")

    sigma_ruido = float(rng.uniform(3.0, 12.0))
    trabajo = trabajo + rng.normal(0.0, sigma_ruido, trabajo.shape).astype("float32")

    resultado = np.clip(trabajo, 0.0, 255.0).astype("uint8")
    sigma_blur = float(rng.uniform(0.0, 1.5))
    if sigma_blur > 0.05:
        resultado = cv2.GaussianBlur(resultado, (0, 0), sigma_blur)
    return resultado


def recortar_casilla(imagen: np.ndarray, matriz_inversa: np.ndarray,
                     fila: int, columna: int) -> np.ndarray:
    """Recorta la casilla respetando la deformacion aplicada a la imagen.

    Si la imagen completa se deformo con ``H``, la casilla que en la vista
    original ocupaba el cuadrado (x0,y0)-(x1,y1) ocupa ahora ``H`` de ese
    cuadrado. Se aplica esa misma transformacion para que el recorte siga
    conteniendo exactamente la misma casilla.
    """
    x0, y0 = columna * CASILLA, fila * CASILLA
    destino = np.array(
        [[0.0, 0.0], [CASILLA - 1.0, 0.0], [CASILLA - 1.0, CASILLA - 1.0],
         [0.0, CASILLA - 1.0]],
        dtype="float32",
    )
    origen = np.array(
        [[x0, y0], [x0 + CASILLA - 1, y0],
         [x0 + CASILLA - 1, y0 + CASILLA - 1], [x0, y0 + CASILLA - 1]],
        dtype="float32",
    )
    matriz = cv2.getPerspectiveTransform(origen, destino) @ matriz_inversa
    return cv2.warpPerspective(
        imagen, matriz, (CASILLA, CASILLA), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def leer_etiquetas(ruta: Path) -> dict[str, dict[int, str]]:
    """{archivo_recorte: {indice_casilla: clase}} solo de las filas 'auto'."""
    if not ruta.exists():
        raise SystemExit(
            f"No encuentro {ruta}. Corre antes autoetiquetar_reales.py"
        )
    posiciones = {casilla(fila, columna): fila * 8 + columna
                  for fila in range(8) for columna in range(8)}
    etiquetas: dict[str, dict[int, str]] = {}
    with open(ruta, "r", encoding="utf-8", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            if fila.get("estado") != "auto":
                continue
            indice = posiciones.get(fila.get("casilla", ""))
            if indice is None:
                continue
            etiquetas.setdefault(fila["archivo_recorte"], {})[indice] = fila["clase"]
    return {nombre: celdas for nombre, celdas in etiquetas.items() if len(celdas) == 64}


def hoja_contacto(variantes: list[np.ndarray], titulo: str) -> np.ndarray:
    """Fila con las 6 variantes y la rejilla dibujada en la primera."""
    mini = 300
    celdas = []
    for indice, imagen in enumerate(variantes):
        vista = cv2.resize(imagen, (mini, mini), interpolation=cv2.INTER_AREA)
        if indice == 0:
            for posicion in range(9):
                p = min(int(round(posicion * mini / 8.0)), mini - 1)
                cv2.line(vista, (p, 0), (p, mini - 1), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.line(vista, (0, p), (mini - 1, p), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(vista, f"v{indice}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vista, f"v{indice}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        celdas.append(vista)
    hoja = np.hstack(celdas)
    banda = np.full((26, hoja.shape[1], 3), 24, dtype="uint8")
    cv2.putText(banda, titulo, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([banda, hoja])


def main(argv=None) -> int:
    analizador = argparse.ArgumentParser(
        description="Genera variantes aumentadas de los recortes reales etiquetados.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analizador.add_argument("--recortes", default=str(RECORTES))
    analizador.add_argument("--etiquetas", default=str(ETIQUETAS))
    analizador.add_argument("--salida", default=str(SALIDA))
    analizador.add_argument("--casillas", default=str(CASILLAS))
    analizador.add_argument("--variantes", type=int, default=VARIANTES,
                            help="variantes por recorte")
    analizador.add_argument("--limite", type=int, default=0,
                            help="procesar solo N recortes (0 = todos)")
    analizador.add_argument("--sin-previews", action="store_true")
    argumentos = analizador.parse_args(argv)

    base = Path(argumentos.recortes)
    etiquetas = leer_etiquetas(Path(argumentos.etiquetas))
    if not etiquetas:
        raise SystemExit(
            f"No hay recortes con 64 casillas 'auto' en {argumentos.etiquetas}. "
            "Corre antes autoetiquetar_reales.py"
        )
    nombres = sorted(etiquetas)
    if argumentos.limite > 0:
        nombres = nombres[:argumentos.limite]

    salida = Path(argumentos.salida)
    carpeta_casillas = Path(argumentos.casillas)
    (salida / "images").mkdir(parents=True, exist_ok=True)
    for clase in CLASES:
        (carpeta_casillas / clase).mkdir(parents=True, exist_ok=True)

    inicio = time.time()
    filas: list[dict] = []
    por_clase = Counter()
    generadas = 0
    casillas_escritas = 0
    muestras: list[tuple[str, list[np.ndarray]]] = []
    faltantes: list[str] = []

    for nombre in nombres:
        origen = base / nombre
        imagen = cv2.imread(str(origen), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if imagen is None:
            faltantes.append(nombre)
            continue
        if imagen.shape[0] != LADO or imagen.shape[1] != LADO:
            imagen = cv2.resize(imagen, (LADO, LADO), interpolation=cv2.INTER_AREA)

        celdas = etiquetas[nombre]
        variantes: list[np.ndarray] = []
        for indice in range(argumentos.variantes):
            rng = np.random.default_rng(_semilla(nombre, indice))
            homografia = homografia_variante(rng)
            deformada = cv2.warpPerspective(
                imagen, homografia, (LADO, LADO), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            variante = fotometria(deformada, rng)
            variantes.append(variante)

            destino = salida / "images" / Path(nombre).parent / (
                f"{Path(nombre).stem}_aug{indice}.jpg"
            )
            destino.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destino), variante, [cv2.IMWRITE_JPEG_QUALITY, 92])
            generadas += 1

            inversa = np.linalg.inv(homografia)
            for indice_casilla in range(64):
                fila, columna = divmod(indice_casilla, 8)
                clase = celdas[indice_casilla]
                recorte = recortar_casilla(variante, inversa, fila, columna)
                archivo_casilla = carpeta_casillas / clase / (
                    f"{Path(nombre).stem}_aug{indice}_{casilla(fila, columna)}.jpg"
                )
                cv2.imwrite(str(archivo_casilla), recorte,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                casillas_escritas += 1
                por_clase[clase] += 1
                filas.append({
                    "archivo_recorte": (Path(nombre).parent / (
                        f"{Path(nombre).stem}_aug{indice}.jpg")).as_posix(),
                    "casilla": casilla(fila, columna),
                    "clase": clase,
                    "fila": fila,
                    "estado": "auto",
                    "motivo": f"aumentado de {nombre}",
                    "variante": indice,
                })
        if len(muestras) < 4:
            muestras.append((nombre, variantes))
        print(f"  {nombre}: {argumentos.variantes} variantes", flush=True)

    csv_salida = salida / "etiquetas_clasificador.csv"
    with open(csv_salida, "w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=CAMPOS)
        escritor.writeheader()
        for fila in filas:
            escritor.writerow(fila)

    previews: list[Path] = []
    if not argumentos.sin_previews and muestras:
        PREVIEWS.mkdir(parents=True, exist_ok=True)
        while len(muestras) < 4:
            muestras.append(muestras[-1])
        for indice, (nombre, variantes) in enumerate(muestras[:4], start=1):
            destino = PREVIEWS / f"augmentado_muestra_{indice}.jpg"
            cv2.imwrite(str(destino), hoja_contacto(variantes, nombre),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            previews.append(destino)

    segundos = time.time() - inicio
    print()
    print(f"listo: {len(nombres)} recortes x {argumentos.variantes} variantes "
          f"en {segundos:.1f} s")
    print(f"  imagenes aumentadas: {generadas} -> {salida / 'images'}")
    print(f"  casillas generadas: {casillas_escritas} -> {carpeta_casillas}")
    print(f"  etiquetas: {csv_salida} ({len(filas)} filas)")
    print(f"  por clase: {dict(sorted(por_clase.items()))}")
    vacias = [clase for clase in CLASES if por_clase.get(clase, 0) == 0]
    if vacias:
        print(f"  AVISO: clases en 0: {vacias}")
    if faltantes:
        print(f"  AVISO: {len(faltantes)} recortes no se pudieron leer: {faltantes[:3]}")
    print(f"  previews: {[p.name for p in previews] if previews else 'omitidas'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
