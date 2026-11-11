"""Autoetiqueta las casillas de los recortes REALES ya rectificados (800x800).

Para cada recorte de ``dataset/reales/recortes/800x800`` se parte el cuadrado en
64 casillas de 100x100, se decide casilla ocupada / vacia por TEXTURA y se
compara la ocupacion con la posicion inicial del ajedrez. Solo se etiqueta
cuando la comparacion es exacta: una foto con una ficha en una casilla que la
posicion inicial no tiene es medio juego disfrazado de inicial (pasa en este
dataset: ``cap1.jpg``) y se marca ``desconocida`` para etiquetarla a mano.

Como se decide ocupada / vacia
------------------------------
Una casilla vacia del tablero impreso es una superficie plana: su gradiente
(Sobel) es casi cero en el interior. Una ficha mete el borde del disco, el
simbolo y su sombra: el gradiente se dispara y, sobre todo, se vuelve
IRREGULAR. Por eso la medida es la desviacion tipica del modulo del gradiente,
no su media: el tablero tiene un gradiente medio bajo pero constante por el
ruido del sensor, mientras que una ficha lo hace variar de un pixel a otro.

El corte se calcula con k-medias de dos grupos sobre las 64 casillas de ESA
foto (la luz cambia mucho entre tomas) y se exige ademas que el grupo alto
supere un piso absoluto: en un tablero vacio las 64 medidas son ruido y
k-medias partiria el ruido por la mitad, inventando 32 fichas. Con el piso,
un tablero vacio da 0 ocupadas.

Se recorta un margen del 8% de cada casilla para no medir la linea que separa
casillas vecinas, que es el borde mas fuerte de una casilla vacia.

Salida
------
``dataset/reales/etiquetas_clasificador.csv`` con una fila por casilla cuando la
foto se puede etiquetar:

    archivo_recorte, casilla, clase, fila, estado, motivo

``archivo_recorte`` es relativo a ``dataset/reales/recortes/800x800``;
``casilla`` va de a1 a h8 (a1 abajo a la izquierda en la vista rectificada);
``clase`` es una de ``classes.txt`` o ``vacia``; ``fila`` es la fila del tablero
(0 = la de las negras, 7 = la de las blancas).

Las fotos que no se etiquetan llevan UNA fila con ``estado`` = ``desconocida``
(ocupacion distinta de la inicial, o colores invertidos) o ``manual``
(``02_medio_juego`` y ``03_finales``: la posicion no se puede deducir).

Uso
---
    python autoetiquetar_reales.py
    python autoetiquetar_reales.py --limite 20 --sin-previews
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
RECORTES = RAIZ / "dataset" / "reales" / "recortes" / "800x800"
CSV_SALIDA = RAIZ / "dataset" / "reales" / "etiquetas_clasificador.csv"
PREVIEWS = RAIZ / "dataset" / "auditoria" / "previews_autolabel"

LADO = 800
CASILLA = LADO // 8
MARGEN = 0.08          # recorte interior para no medir el borde entre casillas
PISO_OCUPADA = 9.0     # contraste minimo (std) para creer que hay ficha
HUECO_MINIMO = 3.0     # hueco minimo entre "plano" y "con ficha"

CAMPOS = ["archivo_recorte", "casilla", "clase", "fila", "estado", "motivo"]

# Posicion inicial: (fila 0..7 desde las negras, columna 0..7 desde la a).
INICIAL: dict[tuple[int, int], str] = {}
PIEZAS_NEGRAS = ["torre_n", "caballo_n", "alfil_n", "dama_n", "rey_n",
                 "alfil_n", "caballo_n", "torre_n"]
PIEZAS_BLANCAS = ["torre_b", "caballo_b", "alfil_b", "dama_b", "rey_b",
                  "alfil_b", "caballo_b", "torre_b"]
for _columna, _clase in enumerate(PIEZAS_NEGRAS):
    INICIAL[(0, _columna)] = _clase
for _columna in range(8):
    INICIAL[(1, _columna)] = "peon_n"
for _columna in range(8):
    INICIAL[(6, _columna)] = "peon_b"
for _columna, _clase in enumerate(PIEZAS_BLANCAS):
    INICIAL[(7, _columna)] = _clase


def casilla(fila: int, columna: int) -> str:
    """(fila, columna) del tablero -> nombre de casilla a1..h8."""
    return f"{chr(ord('a') + columna)}{8 - fila}"


def carpetas_conocidas() -> tuple[set[str], set[str]]:
    """Carpetas cuya posicion se puede deducir sin mirar la foto."""
    return {"01_inicial", "05_tablero_vacio"}, {"02_medio_juego", "03_finales"}


def _leer(ruta: Path) -> np.ndarray | None:
    return cv2.imread(str(ruta), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


def medidas_textura(imagen_bgr: np.ndarray) -> np.ndarray:
    """Contraste (std de intensidad) en cada casilla: 64 valores, fila a fila.

    La medida original era la std del modulo del gradiente. En las fotos de
    640x480, que vienen blandas de foco, el gradiente se aplana y las fichas
    mas desenfocadas caen al mismo nivel que una casilla vacia: medido sobre
    ``capture (11).jpg`` las fichas daban gradiente 13-50 y las vacias 3-8, con
    el corte de k-medias en 17 (dos fichas perdidas). La std de intensidad mide
    lo mismo que importa -- una casilla vacia es una superficie plana -- pero
    conserva la separacion: fichas 13-66, vacias 1-5. El gradiente se sigue
    midiendo para el detalle del CSV.
    """
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (3, 3), 0)
    interior = int(round(MARGEN * CASILLA))
    medidas = np.zeros(64, dtype="float64")
    for fila in range(8):
        for columna in range(8):
            y0 = fila * CASILLA + interior
            y1 = (fila + 1) * CASILLA - interior
            x0 = columna * CASILLA + interior
            x1 = (columna + 1) * CASILLA - interior
            parche = gris[y0:y1, x0:x1]
            medidas[fila * 8 + columna] = float(parche.std())
    return medidas


def medida_gradiente(imagen_bgr: np.ndarray) -> np.ndarray:
    """std del modulo del gradiente por casilla (solo para el detalle)."""
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (3, 3), 0)
    sobel_x = cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gris, cv2.CV_32F, 0, 1, ksize=3)
    magnitud = cv2.magnitude(sobel_x, sobel_y)
    interior = int(round(MARGEN * CASILLA))
    medidas = np.zeros(64, dtype="float64")
    for fila in range(8):
        for columna in range(8):
            parche = magnitud[
                fila * CASILLA + interior:(fila + 1) * CASILLA - interior,
                columna * CASILLA + interior:(columna + 1) * CASILLA - interior,
            ]
            medidas[fila * 8 + columna] = float(parche.std())
    return medidas


def ocupacion(medidas: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Mascara de 64 casillas ocupadas, el corte usado y la separacion.

    El corte es el punto medio del MAYOR hueco entre valores ordenados. k-medias
    de dos grupos no sirve aqui: cuando hay 32 fichas y 32 vacias, los dos grupos
    tienen el mismo tamano y el centro alto cae en medio de la dispersion de las
    fichas, cortando las mas apagadas. Con el mayor hueco el corte cae en la
    separacion real entre "plano" y "con ficha", que en este dataset es de 8-40
    unidades frente a una dispersion interna de 1-5.
    """
    ordenados = np.sort(medidas)
    huecos = np.diff(ordenados)
    posicion = int(np.argmax(huecos))
    hueco = float(huecos[posicion])
    corte = float((ordenados[posicion] + ordenados[posicion + 1]) / 2.0)
    # Un tablero vacio no tiene dos poblaciones: sin hueco grande ni nivel alto,
    # no hay ficha que detectar.
    if hueco < HUECO_MINIMO or float(np.median(ordenados[posicion + 1:])) < PISO_OCUPADA:
        return np.zeros(64, dtype=bool), corte, hueco
    return medidas > corte, corte, hueco


def _brillo_pieza(imagen_bgr: np.ndarray, fila: int, columna: int) -> float:
    """Brillo mediano del centro de la casilla (el cuerpo de la ficha)."""
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    y0 = fila * CASILLA + int(0.25 * CASILLA)
    y1 = fila * CASILLA + int(0.75 * CASILLA)
    x0 = columna * CASILLA + int(0.25 * CASILLA)
    x1 = columna * CASILLA + int(0.75 * CASILLA)
    return float(np.median(gris[y0:y1, x0:x1]))


def _colores_coherentes(imagen_bgr: np.ndarray, ocupadas: np.ndarray) -> tuple[bool, str]:
    """Comprueba que las fichas no esten giradas 180 grados.

    Las fichas de este set: disco NEGRO con simbolo blanco = ``_b`` (blanca) y
    disco BLANCO con simbolo negro = ``_n`` (negra). En la posicion inicial las
    negras van arriba (filas 0-1) y las blancas abajo (filas 6-7), asi que el
    disco de las filas de arriba tiene que ser MAS CLARO que el de abajo. Si sale
    al reves, el recorte esta girado y las etiquetas por posicion tendrian el
    color cambiado.
    """
    arriba = [
        _brillo_pieza(imagen_bgr, fila, columna)
        for fila in range(2) for columna in range(8)
        if ocupadas[fila * 8 + columna]
    ]
    abajo = [
        _brillo_pieza(imagen_bgr, fila, columna)
        for fila in range(6, 8) for columna in range(8)
        if ocupadas[fila * 8 + columna]
    ]
    if len(arriba) < 6 or len(abajo) < 6:
        return False, "faltan fichas en las filas de fondo para comprobar el color"
    if float(np.median(arriba)) <= float(np.median(abajo)):
        return False, ("los discos claros no estan arriba: el recorte esta girado "
                       f"({np.median(arriba):.0f} vs {np.median(abajo):.0f})")
    return True, ""


def _preview(imagen_bgr: np.ndarray, ocupadas: np.ndarray, titulo: str) -> np.ndarray:
    vista = imagen_bgr.copy()
    for indice in range(64):
        fila, columna = divmod(indice, 8)
        color = (0, 0, 255) if ocupadas[indice] else (0, 200, 0)
        cv2.rectangle(vista, (columna * CASILLA + 2, fila * CASILLA + 2),
                      ((columna + 1) * CASILLA - 3, (fila + 1) * CASILLA - 3),
                      color, 2, cv2.LINE_AA)
    cv2.putText(vista, titulo, (8, LADO - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vista, titulo, (8, LADO - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return vista


def _archivos(base: Path) -> list[Path]:
    return sorted(
        ruta for ruta in base.rglob("*")
        if ruta.is_file() and ruta.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def main(argv=None) -> int:
    analizador = argparse.ArgumentParser(
        description="Autoetiqueta las casillas de los recortes reales.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analizador.add_argument("--recortes", default=str(RECORTES),
                            help="carpeta de recortes 800x800")
    analizador.add_argument("--csv", default=str(CSV_SALIDA),
                            help="CSV de etiquetas por casilla")
    analizador.add_argument("--limite", type=int, default=0,
                            help="procesar solo N recortes (0 = todos)")
    analizador.add_argument("--sin-previews", action="store_true",
                            help="no escribir las 5 previews de validacion")
    argumentos = analizador.parse_args(argv)

    base = Path(argumentos.recortes)
    if not base.exists():
        raise SystemExit(f"No encuentro {base}")
    archivos = _archivos(base)
    if argumentos.limite > 0:
        archivos = archivos[:argumentos.limite]
    if not archivos:
        raise SystemExit(f"No hay recortes en {base}")

    inicio = time.time()
    filas: list[dict] = []
    resumen = Counter()
    por_clase = Counter()
    medidas_todas: list[float] = []
    previews: list[Path] = []
    Inicial, Manual = carpetas_conocidas()
    pool_preview: list[tuple[Path, np.ndarray, np.ndarray, str]] = []
    conteo_ocupadas: Counter = Counter()
    gradiente_ocupadas: list[float] = []
    gradiente_vacias: list[float] = []

    for ruta in archivos:
        relativa = ruta.relative_to(base)
        nombre = relativa.as_posix()
        carpeta = relativa.parent.name if relativa.parent != Path(".") else ""
        imagen = _leer(ruta)
        if imagen is None:
            resumen["ilegible"] += 1
            filas.append({"archivo_recorte": nombre, "casilla": "", "clase": "",
                          "fila": "", "estado": "desconocida",
                          "motivo": "no se pudo leer el recorte"})
            continue
        if imagen.shape[0] != LADO or imagen.shape[1] != LADO:
            imagen = cv2.resize(imagen, (LADO, LADO), interpolation=cv2.INTER_AREA)

        medidas = medidas_textura(imagen)
        medidas_todas.extend(medidas.tolist())
        ocupadas, corte, separacion = ocupacion(medidas)
        cuantas = int(ocupadas.sum())
        conteo_ocupadas[cuantas] += 1
        pool_preview.append((ruta, imagen, ocupadas, carpeta))

        if carpeta in Manual:
            resumen["manual"] += 1
            filas.append({
                "archivo_recorte": nombre, "casilla": "", "clase": "",
                "fila": "", "estado": "manual",
                "motivo": (f"{carpeta}: posicion no deducible sin mirar; "
                           f"{cuantas} casillas ocupadas (corte {corte:.1f})"),
            })
            continue

        if carpeta == "05_tablero_vacio":
            if cuantas == 0:
                resumen["auto_vacia"] += 1
                for fila in range(8):
                    for columna in range(8):
                        por_clase["vacia"] += 1
                        filas.append({
                            "archivo_recorte": nombre,
                            "casilla": casilla(fila, columna),
                            "clase": "vacia", "fila": fila, "estado": "auto",
                            "motivo": f"tablero vacio (corte {corte:.1f}, "
                                      f"separacion {separacion:.1f})",
                        })
            else:
                resumen["desconocida"] += 1
                filas.append({
                    "archivo_recorte": nombre, "casilla": "", "clase": "",
                    "fila": "", "estado": "desconocida",
                    "motivo": (f"05_tablero_vacio con {cuantas} casillas ocupadas: "
                               f"o hay algo encima o el corte fallo (corte {corte:.1f})"),
                })
            continue

        # 01_inicial: la ocupacion tiene que ser EXACTAMENTE la inicial.
        esperadas = set(INICIAL)
        obtenidas = {
            (indice // 8, indice % 8) for indice in range(64) if ocupadas[indice]
        }
        if obtenidas != esperadas:
            faltan = len(esperadas - obtenidas)
            sobran = len(obtenidas - esperadas)
            resumen["desconocida"] += 1
            filas.append({
                "archivo_recorte": nombre, "casilla": "", "clase": "",
                "fila": "", "estado": "desconocida",
                "motivo": (f"ocupacion distinta de la inicial: {len(obtenidas)} "
                           f"ocupadas, faltan {faltan}, sobran {sobran}"
                           f" (corte {corte:.1f}, separacion {separacion:.1f})"),
            })
            continue

        coherente, motivo_color = _colores_coherentes(imagen, ocupadas)
        if not coherente:
            resumen["desconocida"] += 1
            filas.append({
                "archivo_recorte": nombre, "casilla": "", "clase": "",
                "fila": "", "estado": "desconocida", "motivo": motivo_color,
            })
            continue

        resumen["auto_inicial"] += 1
        gradiente = medida_gradiente(imagen)
        gradiente_ocupadas.extend(gradiente[ocupadas].tolist())
        gradiente_vacias.extend(gradiente[~ocupadas].tolist())
        for fila in range(8):
            for columna in range(8):
                clase = INICIAL.get((fila, columna), "vacia")
                por_clase[clase] += 1
                filas.append({
                    "archivo_recorte": nombre,
                    "casilla": casilla(fila, columna),
                    "clase": clase, "fila": fila, "estado": "auto",
                    "motivo": (f"posicion inicial exacta (corte {corte:.1f}, "
                               f"separacion {separacion:.1f})"),
                })

    csv_salida = Path(argumentos.csv)
    csv_salida.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_salida, "w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=CAMPOS)
        escritor.writeheader()
        for fila in filas:
            escritor.writerow(fila)

    if not argumentos.sin_previews and pool_preview:
        PREVIEWS.mkdir(parents=True, exist_ok=True)
        # 5 muestras repartidas por la lista: mezcla iniciales, vacios y fallos
        # para que el revisor vea los dos lados del umbral.
        indices = np.linspace(0, len(pool_preview) - 1, num=min(5, len(pool_preview)))
        for indice, posicion in enumerate(indices, start=1):
            ruta, imagen, ocupadas, carpeta = pool_preview[int(round(posicion))]
            titulo = f"{carpeta}/{ruta.stem}  ocupadas={int(ocupadas.sum())}"
            destino = PREVIEWS / f"autolabel_muestra_{indice:02d}.jpg"
            cv2.imwrite(str(destino), _preview(imagen, ocupadas, titulo),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            previews.append(destino)

    segundos = time.time() - inicio
    medidas_todas = np.asarray(medidas_todas)
    total_auto = resumen["auto_inicial"] + resumen["auto_vacia"]
    print()
    print(f"listo: {len(archivos)} recortes en {segundos:.1f} s")
    print(f"  autoetiquetados: {total_auto} "
          f"(inicial {resumen['auto_inicial']}, vacio {resumen['auto_vacia']})")
    print(f"  desconocidos: {resumen['desconocida']}")
    print(f"  manuales (02_medio_juego/03_finales): {resumen['manual']}")
    if resumen["ilegible"]:
        print(f"  ilegibles: {resumen['ilegible']}")
    print(f"  textura: mediana {np.median(medidas_todas):.2f}, "
          f"p90 {np.percentile(medidas_todas, 90):.2f}, "
          f"p99 {np.percentile(medidas_todas, 99):.2f}, "
          f"max {medidas_todas.max():.2f}  (piso {PISO_OCUPADA}, "
          f"hueco minimo {HUECO_MINIMO})")
    if gradiente_ocupadas and gradiente_vacias:
        print(f"  gradiente (control): fichas {np.mean(gradiente_ocupadas):.1f} vs "
              f"vacias {np.mean(gradiente_vacias):.1f}")
    print(f"  casillas por clase: {dict(sorted(por_clase.items()))}")
    vacias_cero = [clase for clase in por_clase if por_clase[clase] == 0]
    if vacias_cero:
        print(f"  AVISO: clases sin ninguna casilla: {vacias_cero}")
    print(f"  etiquetas: {csv_salida}")
    print(f"  previews: {[p.name for p in previews] if previews else 'omitidas'}")
    print(f"  CSV: {len(filas)} filas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
