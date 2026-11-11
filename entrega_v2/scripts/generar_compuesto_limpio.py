"""Compone tableros NUEVOS con los sprites LIMPIOS y el tablero real vacio.

Diferencia con ``compilar_compuestos.py`` (el del lote en cuarentena)
--------------------------------------------------------------------
Aquel tomaba los sprites de ``dataset/reales_compuesto/sprites``, que heredaron
el cruce caballo/alfil de las fotos reales. Este toma los de
``dataset/sprites_limpio``, cortados de la POSICION INICIAL FISICA con etiqueta
por casilla (b8=caballo_n, c8=alfil_n, f8=alfil_n, g8=caballo_n). La clase de
cada ficha es la que la posicion fisica dice, no la que adivino un modelo.

Que produce (misma forma que el lote anterior, carpeta NUEVA)
------------------------------------------------------------
``dataset/compuesto_limpio/images/<nombre>.jpg``      tablero completo
``dataset/compuesto_limpio/labels/<nombre>.txt``      YOLO, 12 clases
``dataset/compuesto_limpio/casillas/clasificador/<clase>/<nombre>_<casilla>.jpg``
                                                      64 recortes de 100x100
``dataset/compuesto_limpio/manifiesto.csv``           fen, perfil, modo, homografia
``dataset/compuesto_limpio/etiquetas_casillas.csv``   imagen, casilla, clase
``dataset/compuesto_limpio/previews/``                muestras anotadas

Reglas
------
* 50 % posiciones LEGALES con ``python-chess`` (partida desde la inicial y N
  jugadas) y 50 % ALEATORIAS con damas/reyes sobrerrepresentados (8-12 por
  tablero): es la clase rara que el detector y el clasificador no ven nunca.
* 70 % perfil ESP 640x480 y 30 % celular, repartido entre 1024x768 y 1280x1024
  (la camara real del usuario ya captura ~1 MP).
* Cada ficha se escala al 83-88 % de la casilla: el mismo tamano que se midio
  en las fotos reales, asi el detector aprende UNA escala y no dos.
* Rotacion de la ficha 0-360 grados, jitter de +-10 % de casilla y sombra
  difusa: el clasificador tolera la ficha girada y el detector no depende de
  que el simbolo mire al frente.
* La FEN de cada tablero se guarda en el manifiesto: las etiquetas se pueden
  verificar contra la posicion declarada, no contra si mismas.

Uso
---
    python generar_compuesto_limpio.py --n 8 --semilla 11 --sin-previews
    python generar_compuesto_limpio.py --n 2500 --semilla 11
"""
from __future__ import annotations

import argparse
import math
import re
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from compilar_compuestos import (
    _colocar,
    _cuotas,
    _fotometria,
    _hoja_casillas,
    _posicion,
    _recortar_casillas,
    _sombra,
    _transferir_luz,
    _variante_tablero,
    _volcar_csv,
    cargar_sprites,
)
from comun import CRUDO, DATASET, cargar_clases
from extraer_sprites import CASILLA, LADO as LADO_PLANO, _estadisticas_cuadrado
from generar_sinteticos import (
    _anotar,
    _linea_yolo,
    _pegar_ficha,
    matriz_a_texto,
)

RAIZ = Path(__file__).resolve().parent.parent
SPRITES = DATASET / "sprites_limpio"
RECORTES = DATASET / "recortes_800"
SALIDA = DATASET / "compuesto_limpio"

# Perfiles de camara. La clave es el sufijo del nombre del archivo.
PERFILES = {
    "esp": dict(
        ancho=640, alto=480,
        relleno=(0.93, 1.00),
        angulo=8.0,
        perspectiva=0.030,
        p_blur=0.92,
        blur=(0.45, 0.95),
        contraste=(1.02, 1.14),
        ganancia=(0.93, 1.05),
        offset=(-14.0, 6.0),
        ruido=(2.0, 6.0),
        jpeg=(62, 82),
    ),
    "celular": dict(
        ancho=1024, alto=768,
        relleno=(0.88, 0.98),
        angulo=8.0,
        perspectiva=0.040,
        p_blur=0.45,
        blur=(0.20, 0.60),
        contraste=(0.98, 1.08),
        ganancia=(0.90, 1.10),
        offset=(-18.0, 14.0),
        ruido=(1.5, 4.5),
        jpeg=(78, 92),
    ),
    "celular_alto": dict(
        ancho=1280, alto=1024,
        relleno=(0.86, 0.97),
        angulo=7.0,
        perspectiva=0.038,
        p_blur=0.40,
        blur=(0.20, 0.55),
        contraste=(0.98, 1.07),
        ganancia=(0.91, 1.09),
        offset=(-16.0, 12.0),
        ruido=(1.2, 4.0),
        jpeg=(80, 93),
    ),
}

# 70 % ESP32 / 30 % celular (15 % + 15 % entre las dos resoluciones de ~1 MP).
CUOTA_PERFIL = {"esp": 0.70, "celular": 0.15, "celular_alto": 0.15}
# 50 % legal / 50 % aleatoria con damas y reyes sobrerrepresentados.
CUOTA_MODO = {"legal": 0.50, "aleatoria": 0.50}

JITTER_CELDA = 0.10          # fraccion de casilla alrededor del centro
ESCALA_FICHA = (0.83, 0.88)  # diametro de la base / lado de casilla (medido)
ESCALA_SOMBRA = (0.10, 0.22)

CAMPOS_MANIFIESTO = [
    "imagen", "fen", "perfil", "modo", "n_fichas", "semilla", "ancho", "alto",
    "fondo", "homografia_recorte",
]
CAMPOS_CASILLAS = ["imagen", "casilla", "clase", "fila", "columna", "perfil"]


def cargar_fondos() -> list[tuple[str, np.ndarray]]:
    """Recortes reales del tablero vacio (800x800)."""
    carpeta = RECORTES / "05_tablero_vacio"
    fondos = []
    for ruta in sorted(carpeta.glob("*.jpg")):
        imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR
                            | cv2.IMREAD_IGNORE_ORIENTATION)
        if imagen is None:
            continue
        if imagen.shape[0] != LADO_PLANO or imagen.shape[1] != LADO_PLANO:
            imagen = cv2.resize(imagen, (LADO_PLANO, LADO_PLANO),
                                interpolation=cv2.INTER_AREA)
        fondos.append((ruta.name, imagen))
    if not fondos:
        raise SystemExit(
            f"No hay recortes de tablero vacio en {carpeta}. "
            "Corre antes rectificar_crudo_v2.py"
        )
    return fondos


def cargar_manteles_limpio(maximo: int = 12) -> list[np.ndarray]:
    """Fotos crudas SIN fichas cerca de las esquinas: finales y tablero vacio.

    ``compilar_compuestos.cargar_manteles`` filtraba por tamano (>=900 px) y
    tomaba cualquier carpeta, incluidas las de 32 fichas: la esquina recortada
    traia FICHAS reales al fondo del compuesto. Una ficha sin etiqueta en el
    fondo es basura para el detector. Las carpetas de finales (2-4 fichas en el
    centro) y tablero vacio (0) tienen las esquinas limpias.
    """
    manteles: list[np.ndarray] = []
    for carpeta in ("04_finales", "05_tablero_vacio"):
        for ruta in sorted((CRUDO / carpeta).glob("*.jpg")):
            imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR
                                | cv2.IMREAD_IGNORE_ORIENTATION)
            if imagen is None:
                continue
            alto, ancho = imagen.shape[:2]
            if min(alto, ancho) < 400:
                continue
            escala = 1200.0 / max(alto, ancho)
            manteles.append(cv2.resize(
                imagen, (int(ancho * escala), int(alto * escala)),
                interpolation=cv2.INTER_AREA,
            ))
            if len(manteles) >= maximo:
                return manteles
    if not manteles:
        raise SystemExit(f"No encontre fotos de mantel en {CRUDO}")
    return manteles


def _fondo_real_limpio(rng: np.random.Generator, ancho: int, alto: int,
                       manteles: list[np.ndarray]) -> np.ndarray:
    """Esquina real de la mesa (mantel/pared), estirada al marco y desenfocada.

    Toma un parche CHICO de la esquina (10-18 % del lado menor): ahi esta la
    mesa, nunca el centro del tablero. El parche se estira al marco, que lo
    desenfoca de forma natural, y se le suma un blur suave. Antes se tomaba
    42-62 % del lado y el "mantel" era el propio damero ampliado: el tablero
    quedaba flotando sobre otro tablero.
    """
    foto = manteles[int(rng.integers(len(manteles)))]
    alto_foto, ancho_foto = foto.shape[:2]
    lado = max(48, int(min(alto_foto, ancho_foto)
                       * float(rng.uniform(0.10, 0.18))))
    x0 = 0 if rng.random() < 0.5 else ancho_foto - lado
    y0 = 0 if rng.random() < 0.5 else alto_foto - lado
    parche = foto[y0:y0 + lado, x0:x0 + lado].copy()
    if rng.random() < 0.5:
        parche = cv2.flip(parche, 1)
    giros = int(rng.integers(4))
    if giros:
        parche = np.ascontiguousarray(np.rot90(parche, giros))

    escala = max(ancho / parche.shape[1], alto / parche.shape[0])
    parche = cv2.resize(
        parche,
        (max(ancho, int(parche.shape[1] * escala) + 1),
         max(alto, int(parche.shape[0] * escala) + 1)),
        interpolation=cv2.INTER_AREA,
    )
    sobra_x = parche.shape[1] - ancho
    sobra_y = parche.shape[0] - alto
    x0 = int(rng.integers(0, sobra_x + 1)) if sobra_x > 0 else 0
    y0 = int(rng.integers(0, sobra_y + 1)) if sobra_y > 0 else 0
    fondo = parche[y0:y0 + alto, x0:x0 + ancho]

    fondo = cv2.GaussianBlur(fondo, (0, 0), float(rng.uniform(1.2, 2.6)))
    imagen = fondo.astype("float32") * float(rng.uniform(0.80, 1.08))
    imagen = imagen + float(rng.uniform(-12.0, 12.0))
    tono = np.array([float(rng.uniform(0.97, 1.03)) for _ in range(3)], "float32")
    return np.clip(imagen * tono, 0, 255).astype(np.uint8)


def fen_de_piezas(piezas: list[tuple[int, int, int]], clases: list[str]) -> str:
    """FEN de la posicion compuesta (turno blanca, sin derechos de enroque).

    Es la verdad declarada del tablero: las etiquetas YOLO se verifican contra
    esta cadena, no contra el compositor.
    """
    import chess

    tablero = chess.Board(None)
    tipos = {"peon": chess.PAWN, "torre": chess.ROOK, "caballo": chess.KNIGHT,
             "alfil": chess.BISHOP, "dama": chess.QUEEN, "rey": chess.KING}
    for fila, columna, clase in piezas:
        forma, _, color = clases[clase].partition("_")
        casilla = chess.square(columna, 7 - fila)
        tablero.set_piece_at(casilla, chess.Piece(tipos[forma], color == "b"))
    tablero.turn = chess.WHITE
    return tablero.fen()


def _siguiente_numero(imagenes_dir: Path) -> dict[str, int]:
    maximo: dict[str, int] = {}
    for ruta in imagenes_dir.glob("compuesto_*.jpg"):
        coincidencia = re.match(r"compuesto_(.+)_(\d+)\.jpg$", ruta.name)
        if coincidencia:
            maximo[coincidencia.group(1)] = max(
                maximo.get(coincidencia.group(1), 0), int(coincidencia.group(2))
            )
    return maximo


def main(argv=None) -> int:
    analizador = argparse.ArgumentParser(
        description="Compone tableros con sprites LIMPIOS y tablero real vacio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analizador.add_argument("--n", type=int, default=2500,
                            help="cantidad de tableros a componer")
    analizador.add_argument("--semilla", type=int, default=11)
    analizador.add_argument("--sprites", default=str(SPRITES))
    analizador.add_argument("--salida", default=str(SALIDA))
    analizador.add_argument("--agregar", action="store_true",
                            help="continuar el dataset: retoma numeracion y "
                                 "conserva las filas de los CSV")
    analizador.add_argument("--sin-casillas", action="store_true",
                            help="no guardar los recortes de 100x100 (NO usar "
                                 "para el paquete de Colab)")
    analizador.add_argument("--previews", type=int, default=3,
                            help="tableros anotados de muestra")
    argumentos = analizador.parse_args(argv)

    if argumentos.n <= 0:
        print("nada que componer: --n debe ser mayor que 0")
        return 2

    clases = cargar_clases()
    if len(clases) != 12:
        raise SystemExit(f"classes.txt debe tener 12 clases y tiene {len(clases)}")

    sprites, todos = cargar_sprites(Path(argumentos.sprites))
    fondos = cargar_fondos()
    manteles = cargar_manteles_limpio()
    print(f"clases: {clases}")
    print(f"sprites: {len(todos)} en {len(sprites)} clases -> {argumentos.sprites}")
    print(f"fondos de tablero real: {[nombre for nombre, _ in fondos]}")
    print(f"fotos de mantel: {len(manteles)}")

    salida = Path(argumentos.salida)
    imagenes_dir = salida / "images"
    etiquetas_dir = salida / "labels"
    casillas_dir = salida / "casillas" / "clasificador"
    previews_dir = salida / "previews"
    imagenes_dir.mkdir(parents=True, exist_ok=True)
    etiquetas_dir.mkdir(parents=True, exist_ok=True)
    (salida / "previews").mkdir(parents=True, exist_ok=True)
    for clase in clases + ["vacia"]:
        (casillas_dir / clase).mkdir(parents=True, exist_ok=True)

    rng_maestro = np.random.default_rng(int(argumentos.semilla))
    perfiles = _cuotas(CUOTA_PERFIL, argumentos.n, rng_maestro)
    modos = _cuotas(CUOTA_MODO, argumentos.n, rng_maestro)
    contadores: Counter = Counter()
    if argumentos.agregar:
        contadores.update(_siguiente_numero(imagenes_dir))
        print(f"--agregar: numeracion retomada {dict(contadores)}")

    inicio = time.time()
    filas_manifiesto: list[dict] = []
    filas_casillas: list[dict] = []
    conteo_perfil: Counter = Counter()
    conteo_modo: Counter = Counter()
    conteo_cajas: Counter = Counter()
    conteo_casillas: Counter = Counter()
    min_ratio = float("inf")
    violaciones = 0
    hojas: dict[str, list[tuple[str, np.ndarray]]] = {}
    anotadas = 0
    muestras: list[Path] = []

    for indice in range(argumentos.n):
        perfil_nombre = perfiles[indice]
        modo = modos[indice]
        perfil = PERFILES[perfil_nombre]
        semilla = int(rng_maestro.integers(0, 2 ** 63))
        rng = np.random.default_rng(semilla)

        nombre_fondo, tablero = fondos[int(rng.integers(len(fondos)))]
        plano = _variante_tablero(tablero, rng)
        piezas = _posicion(rng, clases, modo)

        clases_casilla = [["vacia"] * 8 for _ in range(8)]
        ocupacion: list[tuple[float, float, float, str]] = []
        for fila, columna, clase in piezas:
            clase_nombre = clases[clase]
            opciones = sprites[clase_nombre]
            sprite = opciones[int(rng.integers(len(opciones)))]
            diametro = CASILLA * float(rng.uniform(*ESCALA_FICHA))
            radio_disco = diametro / 2.0
            holgura = max(0.0, 0.95 * (CASILLA / 2.0 - radio_disco))
            jitter = min(JITTER_CELDA * CASILLA, holgura)
            centro_x = (columna + 0.5) * CASILLA + float(
                rng.uniform(-jitter, jitter))
            centro_y = (fila + 0.5) * CASILLA + float(
                rng.uniform(-jitter, jitter))
            angulo = float(rng.uniform(0.0, 360.0))
            destino = _estadisticas_cuadrado(plano, centro_x, centro_y,
                                             radio_disco)
            armonizado = _transferir_luz(
                sprite["imagen"], sprite["fondo"], destino,
                fuerza=float(rng.uniform(0.55, 0.85)),
            )
            _sombra(plano, centro_x, centro_y, diametro, rng)
            _pegar_ficha(plano, armonizado, centro_x, centro_y, diametro, angulo)
            clases_casilla[fila][columna] = clase_nombre
            ocupacion.append((centro_x, centro_y, radio_disco, clase_nombre))

        for i in range(len(ocupacion)):
            for j in range(i + 1, len(ocupacion)):
                distancia = math.hypot(ocupacion[i][0] - ocupacion[j][0],
                                       ocupacion[i][1] - ocupacion[j][1])
                ratio = distancia / max(ocupacion[i][2] + ocupacion[j][2], 1e-6)
                min_ratio = min(min_ratio, ratio)
                if ratio < 0.9:
                    violaciones += 1

        destino_marco = _colocar(LADO_PLANO, LADO_PLANO, perfil["ancho"],
                                 perfil["alto"], rng, perfil, None)
        origen = np.array(
            [[0, 0], [LADO_PLANO, 0], [LADO_PLANO, LADO_PLANO], [0, LADO_PLANO]],
            dtype="float32",
        )
        homografia = cv2.getPerspectiveTransform(origen, destino_marco)
        deformado = cv2.warpPerspective(
            plano, homografia, (perfil["ancho"], perfil["alto"]),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        mascara = cv2.warpPerspective(
            np.full((LADO_PLANO, LADO_PLANO), 255, np.uint8), homografia,
            (perfil["ancho"], perfil["alto"]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mascara = cv2.GaussianBlur(mascara, (0, 0), 0.8).astype("float32") / 255.0
        mantel = _fondo_real_limpio(rng, perfil["ancho"], perfil["alto"], manteles)
        imagen = (
            deformado.astype("float32") * mascara[..., None]
            + mantel.astype("float32") * (1.0 - mascara[..., None])
        )
        imagen = _fotometria(np.clip(imagen, 0, 255).astype(np.uint8), rng, perfil)

        cajas = []
        for centro_x, centro_y, radio, clase_nombre in ocupacion:
            esquinas = np.array([
                [centro_x - radio, centro_y - radio],
                [centro_x + radio, centro_y - radio],
                [centro_x + radio, centro_y + radio],
                [centro_x - radio, centro_y + radio],
            ], dtype="float32")
            proyectadas = cv2.perspectiveTransform(
                esquinas.reshape(-1, 1, 2), homografia
            ).reshape(-1, 2)
            x0 = max(0.0, float(proyectadas[:, 0].min()))
            x1 = min(perfil["ancho"] - 1.0, float(proyectadas[:, 0].max()))
            y0 = max(0.0, float(proyectadas[:, 1].min()))
            y1 = min(perfil["alto"] - 1.0, float(proyectadas[:, 1].max()))
            if x1 - x0 < 2.0 or y1 - y0 < 2.0:
                continue
            cajas.append((x0, y0, x1, y1, clases.index(clase_nombre)))
            conteo_cajas[clase_nombre] += 1

        contadores[perfil_nombre] += 1
        numero = contadores[perfil_nombre]
        nombre = f"compuesto_{perfil_nombre}_{numero:04d}"
        calidad = int(round(float(rng.uniform(*perfil["jpeg"]))))
        cv2.imwrite(str(imagenes_dir / f"{nombre}.jpg"), imagen,
                    [cv2.IMWRITE_JPEG_QUALITY, calidad])

        lineas = [_linea_yolo(caja, perfil["ancho"], perfil["alto"])
                  for caja in cajas]
        with open(etiquetas_dir / f"{nombre}.txt", "w", encoding="utf-8") as archivo:
            for linea in lineas:
                archivo.write(linea + "\n")

        filas_manifiesto.append({
            "imagen": f"{nombre}.jpg",
            "fen": fen_de_piezas(piezas, clases),
            "perfil": perfil_nombre,
            "modo": modo,
            "n_fichas": len(cajas),
            "semilla": semilla,
            "ancho": perfil["ancho"],
            "alto": perfil["alto"],
            "fondo": nombre_fondo,
            "homografia_recorte": matriz_a_texto(np.linalg.inv(homografia)),
        })
        conteo_perfil[perfil_nombre] += 1
        conteo_modo[modo] += 1

        if not argumentos.sin_casillas:
            recortes = _recortar_casillas(imagen, homografia, clases_casilla)
            for posicion, (clase_casilla, recorte) in enumerate(recortes):
                fila, columna = divmod(posicion, 8)
                archivo = casillas_dir / clase_casilla / (
                    f"{nombre}_{chr(ord('a') + columna)}{8 - fila}.jpg")
                cv2.imwrite(str(archivo), recorte,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                filas_casillas.append({
                    "imagen": f"{nombre}.jpg",
                    "casilla": f"{chr(ord('a') + columna)}{8 - fila}",
                    "clase": clase_casilla,
                    "fila": fila,
                    "columna": columna,
                    "perfil": perfil_nombre,
                })
                conteo_casillas[clase_casilla] += 1
            if perfil_nombre not in hojas and len(hojas) < 2:
                hojas[perfil_nombre] = recortes

        if anotadas < argumentos.previews:
            titulo = (f"{nombre}  perfil={perfil_nombre}  modo={modo}  "
                      f"fichas={len(cajas)}")
            destino = previews_dir / f"compuesto_muestra_{anotadas + 1}.jpg"
            cv2.imwrite(str(destino), _anotar(imagen, cajas, clases, titulo),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            muestras.append(destino)
            anotadas += 1

        if (indice + 1) % 100 == 0 or indice + 1 == argumentos.n:
            print(f"  {indice + 1}/{argumentos.n} tableros", flush=True)

    _volcar_csv(salida / "manifiesto.csv", CAMPOS_MANIFIESTO,
                filas_manifiesto, bool(argumentos.agregar))
    _volcar_csv(salida / "etiquetas_casillas.csv", CAMPOS_CASILLAS,
                filas_casillas, bool(argumentos.agregar))

    for indice_hoja, perfil_nombre in enumerate(sorted(hojas)):
        destino = previews_dir / f"casillas_{perfil_nombre}.jpg"
        cv2.imwrite(str(destino), _hoja_casillas(
            hojas[perfil_nombre], f"casillas {perfil_nombre}"),
            [cv2.IMWRITE_JPEG_QUALITY, 90])
        muestras.append(destino)

    segundos = time.time() - inicio
    imagenes = len(list(imagenes_dir.glob("*.jpg")))
    etiquetas = len(list(etiquetas_dir.glob("*.txt")))
    total_cajas = sum(conteo_cajas.values())
    print()
    print(f"listo: {argumentos.n} tableros en {segundos:.1f} s "
          f"({segundos / max(argumentos.n, 1) * 1000:.0f} ms/tablero)")
    print(f"  imagenes: {imagenes} -> {imagenes_dir}")
    print(f"  etiquetas: {etiquetas} ({'OK' if imagenes == etiquetas else 'DESCUADRE'})")
    print(f"  por perfil: {dict(sorted(conteo_perfil.items()))}")
    print(f"  por modo: {dict(sorted(conteo_modo.items()))}")
    print(f"  cajas (total {total_cajas}): {dict(sorted(conteo_cajas.items()))}")
    print(f"  recortes 100x100: {sum(conteo_casillas.values())} "
          f"-> {dict(sorted(conteo_casillas.items()))}")
    print(f"  anti-superposicion: min distancia/suma de radios = {min_ratio:.2f} "
          f"(umbral 0.90), violaciones = {violaciones}")
    print(f"  previews: {[p.name for p in muestras]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
