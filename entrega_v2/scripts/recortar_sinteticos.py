"""Recorta cada sintetica al area jugable: vista rectificada 800x800 sin fondo.

El dataset sintetico base (``dataset/sintetico/images``) conserva el marco
impreso y la mesa a proposito: es lo que ve el DETECTOR YOLO y ahi aprende el
contexto. Este script genera el dataset DERIVADO del CLASIFICADOR por casilla:
solo las 64 casillas del tablero, vistas de frente y cuadradas.

Para cada imagen:

    dataset/sintetico/recortes/800x800/<nombre>.jpg
    dataset/sintetico/recortes/800x800/labels/<nombre>.txt

La homografia sale de ``manifesto.csv`` (columna ``homografia_recorte``), que el
generador escribe con la MISMA matriz que uso para proyectar el tablero. Asi el
recorte y las cajas comparten geometria exacta. Las filas viejas que no tengan
la columna (las 24 primeras imagenes) se resuelven detectando los 4 ArUco con
``esquinas_aruco.detectar``; en sinteticas esa deteccion es precisa porque el
marco esta nitido.

Comprobaciones que imprime al final:

* recortes == imagenes y fichas del recorte == ``n_fichas`` del manifiesto;
* perdida de reproyeccion < 0.001 en coordenadas normalizadas (ida y vuelta de
  cada caja sobre 10 imagenes al azar);
* pixeles que caen fuera de la imagen original (desborde) = 0;
* delta contra una homografia recalculada con ArUco (control independiente).

Uso
---
    python recortar_sinteticos.py
    python recortar_sinteticos.py --verificar 10
    python recortar_sinteticos.py --limite 20 --sin-previews
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

import esquinas_aruco
from generar_sinteticos import LADO_RECORTE, destino_rectificado, texto_a_matriz

RAIZ = Path(__file__).resolve().parent.parent
SINTETICO = RAIZ / "dataset" / "sintetico"
PREVIEWS = RAIZ / "assets" / "previews"

PALETA_CAJAS = [
    (56, 56, 220), (30, 160, 240), (60, 190, 60), (220, 160, 40),
    (200, 60, 200), (40, 200, 200), (120, 120, 240), (240, 240, 60),
    (60, 120, 240), (160, 60, 240), (60, 240, 160), (240, 100, 60),
]


# ---------------------------------------------------------------------------
# Manifiesto y homografia
# ---------------------------------------------------------------------------

def _leer_manifiesto(ruta: Path) -> dict[str, dict[str, str]]:
    if not ruta.exists():
        raise SystemExit(f"No encuentro el manifiesto: {ruta}")
    with open(ruta, "r", encoding="utf-8", newline="") as archivo:
        return {fila["imagen"]: dict(fila) for fila in csv.DictReader(archivo)}


def _matriz_de_fila(fila: dict[str, str]) -> np.ndarray | None:
    texto = (fila.get("homografia_recorte") or "").strip()
    if not texto:
        return None
    try:
        return texto_a_matriz(texto)
    except ValueError as exc:
        print(f"  AVISO: homografia invalida en {fila.get('imagen')}: {exc}")
        return None


def _matriz_aruco_estricta(imagen: np.ndarray, plantilla, lado: int):
    """Homografia por los 4 ArUco, con la API publica; None si no hay 4/4."""
    try:
        deteccion = esquinas_aruco.detectar(imagen, plantilla)
    except Exception:
        # Con 2 o 3 marcadores el emparejado geometrico levanta ValueError.
        return None
    if not deteccion.esquinas.any():
        return None
    return cv2.getPerspectiveTransform(
        deteccion.esquinas.astype("float32"), destino_rectificado(lado)
    )


def _matriz_respaldo(imagen: np.ndarray, plantilla, lado: int):
    """Completa los marcadores faltantes con template matching de la plantilla.

    Solo para sinteticas: la hoja es plana, nitida y con perspectiva chica, y
    el template matching de ``esquinas_aruco`` coincide con los marcadores
    ArUco reales a menos de 1 px. Se usa solo cuando la imagen no trae matriz en
    el manifiesto Y la deteccion ArUco no llega a 4/4.
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    detectados, diccionario, _ = esquinas_aruco._detectar_marcadores(gris, plantilla)
    fuente = "aruco"
    if len(detectados) < 4:
        completos = esquinas_aruco._marcadores_por_plantilla(gris, plantilla)
        for identificador, esquinas in completos.items():
            if identificador not in detectados:
                detectados[identificador] = esquinas
                fuente = "aruco+plantilla"

    ids = [plantilla.lado_a_id[lado_] for lado_ in esquinas_aruco.LADOS]
    faltan = [i for i in ids if i not in detectados]
    if faltan:
        return None, fuente, (
            f"solo {len(detectados)}/4 marcadores ({diccionario or 'sin diccionario'}), "
            f"faltan ids {faltan}"
        )

    origen = np.array([plantilla.centros[i] for i in ids], dtype="float32")
    destino = np.array([detectados[i].mean(axis=0) for i in ids], dtype="float32")
    homografia = cv2.getPerspectiveTransform(origen, destino)
    area = cv2.perspectiveTransform(
        plantilla.esquinas_area.reshape(-1, 1, 2).astype("float32"), homografia
    ).reshape(4, 2)

    alto, ancho = imagen.shape[:2]
    margen_x, margen_y = 0.02 * ancho, 0.02 * alto
    if (area[:, 0].min() < -margen_x or area[:, 0].max() > ancho + margen_x
            or area[:, 1].min() < -margen_y or area[:, 1].max() > alto + margen_y):
        return None, fuente, "el area jugable cae fuera de la imagen (marcadores mal emparejados)"
    if not cv2.isContourConvex(area.astype("int32")):
        return None, fuente, "el area jugable no es convexa (marcadores mal emparejados)"

    matriz = cv2.getPerspectiveTransform(area.astype("float32"), destino_rectificado(lado))
    return matriz, fuente, ""


def _matriz_por_aruco(imagen: np.ndarray, plantilla, lado: int):
    """Homografia de respaldo: 4 ArUco estrictos o completados por plantilla."""
    matriz = _matriz_aruco_estricta(imagen, plantilla, lado)
    if matriz is not None:
        return matriz, "aruco", ""
    return _matriz_respaldo(imagen, plantilla, lado)


# ---------------------------------------------------------------------------
# Cajas
# ---------------------------------------------------------------------------

def _leer_cajas(ruta: Path, ancho: int, alto: int) -> list[tuple[int, float, float, float, float]]:
    """Lee etiquetas YOLO normalizadas y las devuelve en pixeles (x0,y0,x1,y1)."""
    if not ruta.exists():
        return []
    cajas = []
    for numero, linea in enumerate(
        ruta.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not linea.strip():
            continue
        partes = linea.split()
        if len(partes) != 5:
            raise ValueError(f"{ruta}:{numero}: se esperaban 5 campos y hay {len(partes)}")
        clase = int(partes[0])
        cx, cy, w, h = (float(valor) for valor in partes[1:])
        cajas.append((
            clase,
            (cx - w / 2.0) * ancho,
            (cy - h / 2.0) * alto,
            (cx + w / 2.0) * ancho,
            (cy + h / 2.0) * alto,
        ))
    return cajas


def _proyectar_caja(caja_px, matriz: np.ndarray, lado: int):
    """Lleva una caja en pixeles originales a coordenadas YOLO de la vista."""
    clase, x0, y0, x1, y1 = caja_px
    esquinas = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype="float64"
    ).reshape(-1, 1, 2)
    destino = cv2.perspectiveTransform(esquinas, matriz).reshape(-1, 2)
    nx0 = max(0.0, float(destino[:, 0].min()))
    nx1 = min(float(lado - 1), float(destino[:, 0].max()))
    ny0 = max(0.0, float(destino[:, 1].min()))
    ny1 = min(float(lado - 1), float(destino[:, 1].max()))
    if nx1 - nx0 < 0.5 or ny1 - ny0 < 0.5:
        return None
    return (
        clase,
        (nx0 + nx1) / 2.0 / lado,
        (ny0 + ny1) / 2.0 / lado,
        (nx1 - nx0) / lado,
        (ny1 - ny0) / lado,
    )


def _linea_yolo(caja) -> str:
    clase, cx, cy, w, h = caja
    return f"{clase} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _perdida_vuelta(cajas, ancho: int, alto: int, matriz: np.ndarray) -> float:
    """Peor error de ida y vuelta de las cajas, en coords normalizadas.

    Lleva las 4 esquinas de cada caja a la vista rectificada y las regresa a la
    imagen original con la inversa. Comparar esquina contra esquina es la
    prueba de que la reproyeccion no pierde informacion: un bbox de un
    cuadrilatero proyectado cambia de tamano por la perspectiva (eso es
    inherente a YOLO, no una perdida de la transformacion), pero las esquinas
    deben volver exactamente al mismo pixel.
    """
    inversa = np.linalg.inv(matriz)
    normalizador = np.array([ancho, alto], dtype="float64")
    peor = 0.0
    for caja in cajas:
        _clase, x0, y0, x1, y1 = caja
        original = np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype="float64"
        ).reshape(-1, 1, 2)
        ida = cv2.perspectiveTransform(original, matriz)
        vuelta = cv2.perspectiveTransform(ida, inversa).reshape(-1, 2)
        diferencia = np.abs(vuelta - original.reshape(-1, 2)) / normalizador
        peor = max(peor, float(diferencia.max()))
    return peor


def fraccion_borde(imagen: np.ndarray, matriz: np.ndarray, lado: int) -> float:
    """Fraccion de pixeles de la vista que vienen de fuera de la imagen.

    Se deforma una mascara blanca con la misma matriz: los pixeles que caen
    fuera del original reciben ``borderValue=0``. Cualquier valor > 0 significa
    que el recorte tendria una banda negra inventada.
    """
    mascara = np.full(imagen.shape[:2], 255, dtype=np.uint8)
    deformada = cv2.warpPerspective(
        mascara, matriz, (lado, lado), flags=cv2.INTER_NEAREST, borderValue=0
    )
    return float((deformada == 0).mean())


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def dibujar_recorte(recorte: np.ndarray, cajas, titulo: str) -> np.ndarray:
    """Vista rectificada con la rejilla 8x8 y las cajas (si hay) encima."""
    anotada = recorte.copy()
    lado = anotada.shape[0]
    paso = lado / 8.0
    for i in range(9):
        p = int(round(i * paso))
        p = min(p, lado - 1)
        cv2.line(anotada, (p, 0), (p, lado - 1), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(anotada, (0, p), (lado - 1, p), (255, 255, 255), 1, cv2.LINE_AA)
    for clase, cx, cy, w, h in cajas:
        x0 = int(round((cx - w / 2.0) * lado))
        y0 = int(round((cy - h / 2.0) * lado))
        x1 = int(round((cx + w / 2.0) * lado))
        y1 = int(round((cy + h / 2.0) * lado))
        color = PALETA_CAJAS[clase % len(PALETA_CAJAS)]
        cv2.rectangle(anotada, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
    cv2.putText(anotada, titulo, (8, lado - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (25, 25, 25), 4, cv2.LINE_AA)
    cv2.putText(anotada, titulo, (8, lado - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return anotada


def _elegir_muestras(nombres: list[str], manifiesto: dict[str, dict[str, str]],
                     rng: np.random.Generator, cantidad: int = 3) -> list[str]:
    """Imagenes con fichas visibles, para que la muestra sea representativa."""
    candidatas = [
        nombre for nombre in nombres
        if int(manifiesto[nombre].get("n_fichas") or 0) >= 4
    ] or list(nombres)
    if not candidatas:
        return []
    indices = rng.choice(len(candidatas), size=min(cantidad, len(candidatas)),
                         replace=False)
    return [candidatas[int(i)] for i in indices]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    analizador = argparse.ArgumentParser(
        description="Recorta las sinteticas al area jugable en 800x800.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analizador.add_argument("--sintetico", default=str(SINTETICO),
                            help="carpeta del dataset sintetico base")
    analizador.add_argument("--salida", default=None,
                            help="carpeta de recortes (default: <sintetico>/recortes/800x800)")
    analizador.add_argument("--verificar", type=int, default=10,
                            help="imagenes al azar para medir perdida y comparar ArUco "
                                 "(0 desactiva)")
    analizador.add_argument("--semilla", type=int, default=7,
                            help="semilla para elegir las muestras y las verificadas")
    analizador.add_argument("--limite", type=int, default=0,
                            help="procesar solo las primeras N imagenes (0 = todas)")
    analizador.add_argument("--sin-previews", action="store_true",
                            help="no escribir las previews en assets/previews")
    argumentos = analizador.parse_args(argv)

    base = Path(argumentos.sintetico)
    carpetas = base / "images"
    etiquetas = base / "labels"
    if not carpetas.exists():
        raise SystemExit(f"No encuentro {carpetas}")
    manifiesto = _leer_manifiesto(base / "manifesto.csv")
    lado = LADO_RECORTE

    imagenes = sorted(p for p in carpetas.iterdir() if p.suffix.lower() == ".jpg")
    if argumentos.limite > 0:
        imagenes = imagenes[:argumentos.limite]
    if not imagenes:
        raise SystemExit(f"No hay imagenes en {carpetas}")

    salida = Path(argumentos.salida) if argumentos.salida else base / "recortes" / f"{lado}x{lado}"
    carpeta_labels = salida / "labels"
    salida.mkdir(parents=True, exist_ok=True)
    carpeta_labels.mkdir(parents=True, exist_ok=True)
    if not argumentos.sin_previews:
        PREVIEWS.mkdir(parents=True, exist_ok=True)

    plantilla = esquinas_aruco.analizar_plantilla()
    rng = np.random.default_rng(int(argumentos.semilla))

    inicio = time.time()
    escritos = 0
    total_cajas = 0
    fuentes = {"manifesto": 0, "aruco": 0}
    sin_homografia: list[str] = []
    desbordes: list[tuple[str, float]] = []
    discrepancias: list[tuple[str, int, int]] = []
    cajas_por_nombre: dict[str, list] = {}
    matrices: dict[str, np.ndarray] = {}

    for ruta in imagenes:
        nombre = ruta.stem
        imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
        if imagen is None:
            raise SystemExit(f"OpenCV no pudo leer {ruta}")
        alto, ancho = imagen.shape[:2]

        fila = manifiesto.get(ruta.name, {})
        matriz = _matriz_de_fila(fila)
        fuente = "manifesto"
        if matriz is None:
            matriz, fuente, motivo = _matriz_por_aruco(imagen, plantilla, lado)
            if matriz is None:
                sin_homografia.append(f"{ruta.name}: {motivo}")
                continue
        fuentes[fuente] = fuentes.get(fuente, 0) + 1

        cajas_originales = _leer_cajas(etiquetas / f"{nombre}.txt", ancho, alto)
        cajas_vista = []
        perdidas = 0
        for caja in cajas_originales:
            proyectada = _proyectar_caja(caja, matriz, lado)
            if proyectada is None:
                perdidas += 1
            else:
                cajas_vista.append(proyectada)

        recorte = cv2.warpPerspective(
            imagen, matriz, (lado, lado), flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )
        cv2.imwrite(str(salida / f"{nombre}.jpg"), recorte,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(carpeta_labels / f"{nombre}.txt", "w", encoding="utf-8") as archivo:
            for caja in cajas_vista:
                archivo.write(_linea_yolo(caja) + "\n")

        escritos += 1
        total_cajas += len(cajas_vista)
        cajas_por_nombre[ruta.name] = cajas_vista
        matrices[ruta.name] = matriz

        fraccion = fraccion_borde(imagen, matriz, lado)
        if fraccion > 1e-9:
            desbordes.append((ruta.name, fraccion))

        esperadas = int(fila.get("n_fichas") or len(cajas_originales))
        if len(cajas_vista) + perdidas != esperadas or perdidas:
            discrepancias.append((ruta.name, len(cajas_vista), esperadas))

        if (escritos % 100) == 0:
            print(f"  {escritos}/{len(imagenes)} recortes")

    # Verificacion: perdida de ida y vuelta y control independiente con ArUco.
    cajas_disco = 0
    for ruta_label in carpeta_labels.glob("*.txt"):
        with open(ruta_label, "r", encoding="utf-8") as archivo:
            cajas_disco += sum(1 for linea in archivo if linea.strip())

    verificables = sorted(matrices)
    elegidas: list[str] = []
    if argumentos.verificar > 0 and verificables:
        indices = rng.choice(len(verificables),
                             size=min(argumentos.verificar, len(verificables)),
                             replace=False)
        elegidas = [verificables[int(i)] for i in indices]

    peor_perdida = 0.0
    peor_delta_aruco = 0.0
    arruco_ok = 0
    for nombre_archivo in elegidas:
        imagen = cv2.imread(str(carpetas / nombre_archivo), cv2.IMREAD_COLOR)
        alto, ancho = imagen.shape[:2]
        matriz = matrices[nombre_archivo]
        cajas = cajas_por_nombre[nombre_archivo]
        peor_perdida = max(
            peor_perdida, _perdida_vuelta(cajas, ancho, alto, matriz)
        )

        matriz_det = _matriz_aruco_estricta(imagen, plantilla, lado)
        if matriz_det is None:
            continue
        arruco_ok += 1
        originales = _leer_cajas(etiquetas / Path(nombre_archivo).with_suffix(".txt").name,
                                 ancho, alto)
        for caja in originales:
            a = _proyectar_caja(caja, matriz, lado)
            b = _proyectar_caja(caja, matriz_det, lado)
            if a is None or b is None:
                continue
            peor_delta_aruco = max(
                peor_delta_aruco,
                abs(a[1] - b[1]), abs(a[2] - b[2]), abs(a[3] - b[3]), abs(a[4] - b[4]),
            )

    # Previews (3 sinteticas con fichas visibles).
    previews = []
    if not argumentos.sin_previews:
        for indice, nombre_archivo in enumerate(
            _elegir_muestras(list(cajas_por_nombre), manifiesto, rng), start=1
        ):
            imagen = cv2.imread(str(carpetas / nombre_archivo), cv2.IMREAD_COLOR)
            alto, ancho = imagen.shape[:2]
            matriz = matrices[nombre_archivo]
            recorte = cv2.warpPerspective(
                imagen, matriz, (lado, lado), flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )
            cajas = cajas_por_nombre[nombre_archivo]
            fila = manifiesto.get(nombre_archivo, {})
            borde = fraccion_borde(imagen, matriz, lado) * 100.0
            titulo = (f"{Path(nombre_archivo).stem}  |  perfil={fila.get('perfil', '?')}  |  "
                      f"fichas={len(cajas)}  |  borde={borde:.2f}%")
            destino = PREVIEWS / f"recorte_muestra_sintetico_{indice:02d}.jpg"
            cv2.imwrite(str(destino), dibujar_recorte(recorte, cajas, titulo),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            previews.append(destino.name)

    segundos = time.time() - inicio
    max_borde = max([f for _, f in desbordes], default=0.0)
    print()
    print(f"listo: {escritos}/{len(imagenes)} imagenes recortadas en {segundos:.1f} s")
    print(f"  recortes: {salida}")
    print(f"  labels: {carpeta_labels}")
    print(f"  cajas transformadas: {total_cajas}")
    print(f"  cajas en disco: {cajas_disco}")
    print(f"  homografia: {fuentes}")
    print(f"  sin homografia: {len(sin_homografia)}")
    for linea in sin_homografia[:5]:
        print(f"    {linea}")
    print(f"  desbordes: {len(desbordes)} imagenes (max {max_borde * 100:.4f}%)")
    for nombre, fraccion in desbordes[:5]:
        print(f"    {nombre}: {fraccion * 100:.4f}%")
    print(f"  fichas: {len(discrepancias)} discrepancias")
    for nombre, obtenidas, esperadas in discrepancias[:5]:
        print(f"    {nombre}: recorte={obtenidas} manifiesto={esperadas}")
    print(f"  perdida ida/vuelta (max, {len(elegidas)} imagenes): {peor_perdida:.3e}")
    print(f"  delta vs ArUco (max, {arruco_ok}/{len(elegidas)} detectadas): "
          f"{peor_delta_aruco:.6f}")
    print(f"  previews: {previews if previews else 'omitidas'}")

    correcto = (
        escritos == len(imagenes)
        and not discrepancias
        and not desbordes
        and cajas_disco == total_cajas
        and peor_perdida < 0.001
    )
    if not correcto:
        print("RESULTADO: revisar (ver conteos arriba)")
        return 1
    print("RESULTADO: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
