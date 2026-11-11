"""Utilidades compartidas por los scripts del proyecto.

La llave de Roboflow NUNCA vive en el codigo: se lee de la variable de entorno
ROBOFLOW_API_KEY.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATASET = RAIZ / "dataset"
CRUDO = DATASET / "crudo"
EXPORTS = RAIZ / "exports"
CALIBRACION = RAIZ / "calibracion"
CLASES_TXT = RAIZ / "classes.txt"

# Proyectos hermanos: v1 queda intacta como referencia y el repo del companero
# aporta homografia, FEN y captura. Se leen, nunca se editan.
V1 = RAIZ.parent / "ajedrez-yolo"
EQUIPO = RAIZ.parent / "ajedrez-yolo-equipo"

WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "tav0workspaces")
PROYECTO = os.environ.get("ROBOFLOW_PROJECT", "ajedrez-fichas")

# El orden importa: define el sufijo de la tecla de captura y el batch en Roboflow.
CARPETAS = [
    "01_inicial_intacta",
    "02_inicial_con_jugadas",
    "03_medio_juego",
    "04_finales",
    "05_tablero_vacio",
]

# Cuantas cajas debe tener CADA foto de la carpeta. None = sin restriccion.
# 01_inicial_intacta: tablero integro fisico = 32 piezas.
# 02_inicial_con_jugadas: apertura con 2-6 jugadas (el usuario declara FEN).
CAJAS_ESPERADAS: dict[str, int | None] = {
    "01_inicial_intacta": 32,
    "02_inicial_con_jugadas": None,
    "03_medio_juego": None,
    "04_finales": None,
    "05_tablero_vacio": 0,
}

EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# classes.txt de v2 esta en espanol (peon_b..rey_n), pero los modelos
# disponibles predicen en ingles. Traducir en el borde evita que el resto del
# pipeline tenga que conocer las dos convenciones.
FORMAS_ES = {
    "peon": "pawn",
    "torre": "rook",
    "caballo": "knight",
    "alfil": "bishop",
    "dama": "queen",
    "rey": "king",
}


def cargar_clases() -> list[str]:
    """Lee classes.txt ignorando comentarios y lineas vacias."""
    if not CLASES_TXT.exists():
        raise SystemExit(f"No encuentro {CLASES_TXT}")
    clases = [
        linea.strip()
        for linea in CLASES_TXT.read_text(encoding="utf-8-sig").splitlines()
        if linea.strip() and not linea.lstrip().startswith("#")
    ]
    if len(set(clases)) != len(clases):
        raise SystemExit("classes.txt tiene clases duplicadas")
    if not clases:
        raise SystemExit("classes.txt esta vacio")
    return clases


def api_key() -> str:
    clave = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not clave:
        raise SystemExit(
            "Falta la variable ROBOFLOW_API_KEY.\n"
            'En PowerShell:  $env:ROBOFLOW_API_KEY = "tu_llave_privada"'
        )
    return clave


def imagenes(carpeta: Path) -> list[Path]:
    """Imagenes de una carpeta, ordenadas por nombre. No es recursivo."""
    if not carpeta.exists():
        return []
    return sorted(
        p for p in carpeta.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONES
    )


def carpeta_de(nombre: str) -> str | None:
    """Deduce la carpeta de origen a partir del nombre del archivo."""
    for carpeta in CARPETAS:
        if nombre.startswith(carpeta):
            return carpeta
    return None


def modelo_por_defecto() -> Path:
    """Mejor modelo disponible sin obligar a pasar --modelo.

    Prioriza el best.pt final de v1: es el unico entrenado con fotos reales.
    Si falta, sigue con el run mas reciente y despues con el sintetico que ya
    vive en modelos/. Asi los scripts corren hoy y en T5 solo cambia el archivo.
    """
    candidatos = [V1 / "models" / "best.pt"]
    candidatos += sorted(
        (V1 / "runs").glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    candidatos += [
        V1 / "modelo_v2_girado.pt",
        RAIZ / "modelos" / "best.pt",
        RAIZ / "modelos" / "modelo_sintetico.pt",
    ]
    for candidato in candidatos:
        if candidato.exists():
            return candidato
    return RAIZ / "modelos" / "best.pt"


def normalizar_clase(clase: str) -> str:
    """Traduce peon_b -> white_pawn. Deja intacto lo que ya viene en ingles."""
    if clase.startswith(("white_", "black_")):
        return clase
    forma, _, sufijo = clase.rpartition("_")
    if sufijo in ("b", "n") and forma in FORMAS_ES:
        return f"{'white' if sufijo == 'b' else 'black'}_{FORMAS_ES[forma]}"
    return clase


def normalizar_detecciones(detecciones):
    """Aplica normalizar_clase a (clase, cx, cy, conf) sin tocar el resto."""
    return [(normalizar_clase(d[0]),) + tuple(d[1:]) for d in detecciones]


def cargar_modulos_equipo():
    """Importa board/detect/esp32 del repo del companero sin duplicar su codigo."""
    sys.path.insert(0, str(EQUIPO / "src"))
    import board  # noqa: F401
    import detect  # noqa: F401
    import esp32  # noqa: F401

    return board, detect, esp32


def kmedias_1d(valores, k: int, iteraciones: int = 80):
    """Agrupa valores en k centros por k-medias.

    No uso agrupado secuencial con tolerancia: si los valores son densos, cada
    uno cae dentro de la tolerancia del anterior y todos acaban en un solo grupo
    por efecto cadena. Eso fue justo lo que devolvio "4 columnas, 2 filas" con
    una rejilla de 8x4.
    """
    import numpy as np

    datos = np.asarray(valores, dtype="float64")
    ordenados = np.sort(datos)
    centros = np.quantile(ordenados, [(i + 0.5) / k for i in range(k)])
    for _ in range(iteraciones):
        etiquetas = np.abs(datos[:, None] - centros[None, :]).argmin(axis=1)
        nuevos = centros.copy()
        for j in range(k):
            if np.any(etiquetas == j):
                nuevos[j] = datos[etiquetas == j].mean()
        if np.allclose(nuevos, centros, atol=0.01):
            centros = nuevos
            break
        centros = nuevos
    return np.sort(centros)


def _validar_rejilla(puntos, esquinas, tamano: int = 800) -> str | None:
    """Comprueba que las piezas caen en casillas distintas y centradas.

    Es la validacion que importa: no que las esquinas se vean bonitas, sino que
    cada pieza acabe en una casilla propia. Si dos piezas comparten casilla o
    alguna se sale del tablero, la homografia esta mal y hay que rechazarla en
    vez de devolver un FEN invalido.
    """
    import cv2
    import numpy as np

    destino = np.array([
        [0, 0], [tamano - 1, 0], [tamano - 1, tamano - 1], [0, tamano - 1]
    ], dtype="float32")
    matriz = cv2.getPerspectiveTransform(esquinas, destino)
    proyectados = cv2.perspectiveTransform(puntos.reshape(-1, 1, 2), matriz).reshape(-1, 2)

    lado = tamano / 8.0
    columnas = (proyectados[:, 0] // lado).astype(int)
    filas = (proyectados[:, 1] // lado).astype(int)
    dentro = (columnas >= 0) & (columnas < 8) & (filas >= 0) & (filas < 8)
    if dentro.sum() < len(puntos) * 0.85:
        return (
            f"{int((~dentro).sum())} de {len(puntos)} piezas caen fuera del tablero: "
            "la rejilla esta mal centrada"
        )

    # Un par de detecciones fuera o repetidas son falsos positivos; una mayoria
    # repetida significa que la homografia esta corrida.
    ocupadas = set(zip(columnas[dentro].tolist(), filas[dentro].tolist()))
    repetidas = int(dentro.sum()) - len(ocupadas)
    if repetidas > 2:
        return f"{repetidas} piezas comparten casilla"

    # Cada pieza debe estar cerca del centro de su casilla, no en el borde.
    resto_x = (proyectados[dentro, 0] % lado) / lado - 0.5
    resto_y = (proyectados[dentro, 1] % lado) / lado - 0.5
    desvio = float(np.max(np.hypot(resto_x, resto_y)))
    if desvio > 0.35:
        return f"las piezas quedan descentradas ({desvio:.2f} casillas)"
    return None


def filtrar_aisladas(puntos, factor: float = 1.8, pasadas: int = 3):
    """Quita detecciones sin vecinas: son falsos positivos, no piezas.

    Las piezas de un tablero estan a un paso de casilla o a 1.41 del vecino mas
    cercano. Un falso positivo sobre el cable o el telefono queda solo. Sin
    filtrarlos, dos cajas perdidas en una esquina del encuadre arruinan las
    estadisticas de columnas (72 px de paso donde hay 46) y la rejilla se
    rechaza sola.

    Devuelve (puntos_filtrados, cuantos_se_quitaron).
    """
    import numpy as np

    datos = np.asarray(puntos, dtype="float64")
    quitados = 0
    for _ in range(pasadas):
        if len(datos) < 4:
            break
        distancias = np.linalg.norm(datos[:, None, :] - datos[None, :, :], axis=2)
        np.fill_diagonal(distancias, np.inf)
        vecino = distancias.min(axis=1)
        limite = factor * float(np.median(vecino))
        quedan = vecino <= limite
        if quedan.all():
            break
        quitados += int((~quedan).sum())
        datos = datos[quedan]
    return datos, quitados


def _ajustar_homografia(puntos, tamano: int = 800, iteraciones: int = 4):
    """Ajusta la homografia cenital->imagen usando TODAS las piezas.

    Arranca con las cuatro piezas de esquina, que en la posicion inicial estan
    en el centro de las casillas a1, h1, a8 y h8. Con esa estimacion proyecta
    cada pieza a la vista cenital, la asigna a la casilla mas cercana y reajusta
    la homografia por minimos cuadrados con RANSAC.

    Es lo que hace que el resultado no dependa de acertar las cuatro esquinas:
    si una falta o hay un falso positivo, RANSAC lo descarta y quedan ~30
    correspondencias reales. Con solo 4 puntos un error de una pieza corria
    media casilla toda la rejilla.

    Devuelve (matriz | None, motivo).
    """
    import cv2
    import numpy as np

    lado = tamano / 8.0
    suma = puntos[:, 0] + puntos[:, 1]
    diferencia = puntos[:, 0] - puntos[:, 1]
    extremos = [int(np.argmin(suma)), int(np.argmax(diferencia)),
                int(np.argmax(suma)), int(np.argmin(diferencia))]
    if len(set(extremos)) != 4:
        return None, "las cuatro piezas de esquina no son puntos distintos"

    centros = np.array([
        [lado / 2, lado / 2],
        [tamano - lado / 2, lado / 2],
        [tamano - lado / 2, tamano - lado / 2],
        [lado / 2, tamano - lado / 2],
    ], dtype="float32")
    matriz = cv2.getPerspectiveTransform(centros, puntos[extremos].astype("float32"))

    inliers = 0
    for _ in range(iteraciones):
        try:
            inversa = np.linalg.inv(matriz)
        except np.linalg.LinAlgError:
            return None, "la homografia inicial es degenerada"

        cenital = cv2.perspectiveTransform(
            puntos.reshape(-1, 1, 2), inversa
        ).reshape(-1, 2)

        columna = np.round(cenital[:, 0] / lado - 0.5)
        fila = np.round(cenital[:, 1] / lado - 0.5)
        residuo = np.hypot(
            cenital[:, 0] - (columna + 0.5) * lado,
            cenital[:, 1] - (fila + 0.5) * lado,
        )
        valido = (
            (columna >= 0) & (columna < 8) & (fila >= 0) & (fila < 8)
            & (residuo < 0.45 * lado)
        )
        inliers = int(valido.sum())
        if inliers < 6:
            return None, f"solo {inliers} piezas encajan en la rejilla"

        origen = np.stack(
            [(columna[valido] + 0.5) * lado, (fila[valido] + 0.5) * lado], axis=1
        ).astype("float32")
        nueva, _ = cv2.findHomography(origen, puntos[valido], cv2.RANSAC, 3.0)
        if nueva is None:
            return None, "RANSAC no encontro homografia"
        matriz = nueva

    return matriz, f"{inliers} piezas ajustadas"


def esquinas_por_rejilla(detecciones, tamano: int = 800):
    """Esquinas del tablero a partir de las piezas, sin buscar bordes.

    La camara ve el tablero en perspectiva: es un trapecio, no un cuadrado
    girado. Ajustar un cuadrado con paso constante deja piezas fuera en cuanto
    la perspectiva se nota; una homografia de 4 puntos si la modela.

    El punto de partida son las cuatro piezas de esquina (a1, h1, a8, h8 en la
    posicion inicial), que estan exactamente en el centro de su casilla. Desde
    ahi se refinan con todas las piezas, asi que el resultado sobrevive a una
    esquina perdida o a un falso positivo.

    Devuelve (esquinas | None, motivo).
    """
    import cv2
    import numpy as np

    if len(detecciones) < 8:
        return None, f"muy pocas detecciones ({len(detecciones)}) para inferir la rejilla"

    puntos = np.array([[d[1], d[2]] for d in detecciones], dtype="float32")
    puntos, quitadas = filtrar_aisladas(puntos)
    puntos = puntos.astype("float32")
    if quitadas:
        print(f"  {quitadas} detecciones aisladas descartadas como falsos positivos")
    if len(puntos) < 8:
        return None, f"quedan muy pocas piezas ({len(puntos)})"

    matriz, motivo = _ajustar_homografia(puntos, tamano)
    if matriz is None:
        return None, motivo

    bordes = np.array([
        [[0.0, 0.0]], [[tamano - 1.0, 0.0]],
        [[tamano - 1.0, tamano - 1.0]], [[0.0, tamano - 1.0]],
    ], dtype="float32")
    esquinas = cv2.perspectiveTransform(bordes, matriz).reshape(4, 2).astype("float32")

    problema = _validar_rejilla(puntos, ordenar_esquinas(esquinas), tamano)
    if problema is not None:
        return None, f"rejilla descartada: {problema}"

    ancho = float(np.linalg.norm(esquinas[1] - esquinas[0])) / 8
    return esquinas, f"{ancho:.0f} px por casilla, {motivo}"


def ordenar_esquinas(esquinas):
    """TL, TR, BR, BL. Misma logica que board.ordenar_esquinas."""
    import numpy as np

    puntos = np.asarray(esquinas, dtype="float32").reshape(4, 2)
    suma = puntos.sum(axis=1)
    diferencia = np.diff(puntos, axis=1).reshape(-1)
    return np.array([
        puntos[np.argmin(suma)],
        puntos[np.argmin(diferencia)],
        puntos[np.argmax(suma)],
        puntos[np.argmax(diferencia)],
    ], dtype="float32")


def corregir_color(gris, detecciones, radio: int, umbral: float = 0.09,
                   factor: float = 0.55):
    """Reemplaza el color que dijo el modelo por el medido en la imagen.

    El color lo da el RELLENO del simbolo, no la base de la ficha: contorno
    fino = blanca, relleno macizo = negra. Es una senal de un solo bit y el
    modelo entrenado la confunde casi siempre (en las fotos del dataset el
    relleno salia lavado por la exposicion). Con las fichas nuevas de base
    blanca/negra se mantiene esta medida como verificacion.

    Medido sobre pixeles la separacion es total: blancas <= 0.01 de fraccion
    oscura, negras >= 0.18. El corte no es fijo: se calcula por k-medias con dos
    grupos sobre las piezas del propio frame. Un umbral fijo dejaba pasar una
    ficha en el borde del tablero, donde la sombra del borde oscurece el parche;
    el corte auto-calibrado la recupera sin recalibrar a mano.

    Devuelve (detecciones_corregidas, cuantas_cambiaron).
    """
    import numpy as np

    def medir(deteccion):
        x, y = int(round(deteccion[1])), int(round(deteccion[2]))
        parche = gris[max(y - radio, 0):y + radio, max(x - radio, 0):x + radio]
        if parche.size < 20:
            return None
        base = float(np.percentile(parche, 90))
        if base < 30:  # ficha casi negra: no hay contraste para decidir
            return None
        return float((parche < base * factor).mean())

    rellenos = [medir(d) for d in detecciones]

    # Corte auto-calibrado: si el frame tiene los dos colores, sus fracciones
    # forman dos grupos bien separados y el punto medio es el mejor umbral.
    frontera = umbral
    medidos = [r for r in rellenos if r is not None]
    if len(medidos) >= 4:
        centros = kmedias_1d(medidos, 2)
        if len(centros) == 2 and (centros[1] - centros[0]) > 0.06:
            frontera = float((centros[0] + centros[1]) / 2)

    corregidas = []
    cambios = 0
    for deteccion, relleno in zip(detecciones, rellenos):
        prefijo, _, forma = deteccion[0].partition("_")
        if prefijo not in ("white", "black") or not forma or relleno is None:
            corregidas.append(deteccion)
            continue
        color = "black" if relleno >= frontera else "white"
        cambios += int(color != prefijo)
        corregidas.append((f"{color}_{forma}",) + tuple(deteccion[1:]))

    return corregidas, cambios


def radio_estimado(detecciones, minimo: int = 5) -> int:
    """Radio en pixeles para medir el color, sin conocer aun las esquinas.

    La distancia entre piezas vecinas es el paso de casilla, y el simbolo ocupa
    cerca de la mitad. Con eso la medida del color funciona antes de tener la
    homografia, que es justo lo que hace falta para decidir el giro de 180.
    """
    import numpy as np

    if len(detecciones) < 2:
        return minimo
    puntos = np.array([[d[1], d[2]] for d in detecciones], dtype="float64")
    distancias = np.linalg.norm(puntos[:, None, :] - puntos[None, :, :], axis=2)
    np.fill_diagonal(distancias, np.inf)
    vecino = float(np.median(distancias.min(axis=1)))
    if not np.isfinite(vecino) or vecino <= 0:
        return minimo
    return max(minimo, int(round(vecino * 0.2)))


def necesita_giro(detecciones) -> bool:
    """True si las blancas estan arriba y hay que girar el frame 180 grados.

    `construir_fen` hace `fila = 7 - fila_visual`: la fila visual de arriba es la
    8. Las blancas van en las filas 1-2, es decir abajo en la imagen. Si salen
    arriba, el FEN queda con los colores invertidos sin dar ningun error.
    """
    blancas = [d[2] for d in detecciones if d[0].startswith("white_")]
    negras = [d[2] for d in detecciones if d[0].startswith("black_")]
    if not blancas or not negras:
        return False
    return (sum(blancas) / len(blancas)) < (sum(negras) / len(negras))
