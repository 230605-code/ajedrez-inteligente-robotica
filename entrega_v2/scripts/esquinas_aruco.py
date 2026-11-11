"""Deteccion de las 4 esquinas del AREA JUGABLE con marcadores ArUco.

Los marcadores van FUERA del area jugable. Los IDs y su ubicacion (TL/TR/BR/BL)
no se codifican a mano: se aprenden una sola vez de ``plantilla/tablero.png``
ejecutando la misma deteccion sobre la plantilla. Si cambia la plantilla o el
diccionario, la siguiente corrida lo vuelve a deducir.

Idea geometrica
---------------
La plantilla y una foto son dos proyecciones del MISMO plano fisico, asi que
existe una homografia entre ambas. Con los 4 centros de los marcadores como
correspondencias se calcula esa homografia y se aplica a las esquinas del area
jugable medidas en la plantilla. De esa forma las esquinas de la foto se derivan
aunque los marcadores esten fuera del tablero y la camara vea perspectiva.

Uso
---
    python esquinas_aruco.py
    python esquinas_aruco.py --imagen foto.jpg --debug-salida salida.jpg

La funcion publica es ``detectar(fuente)`` y devuelve ``Deteccion`` con las
esquinas del area jugable (TL, TR, BR, BL), los IDs detectados y la confianza.
``dibujar_debug(fuente)`` pinta esquinas, lados, IDs y un resumen de texto.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
PLANTILLA_TABLERO = RAIZ / "plantilla" / "tablero.png"

LADOS = ("TL", "TR", "BR", "BL")

# Se prueban en orden; el primero que devuelva los 4 marcadores gana. 4x4 es el
# que trae la plantilla actual, el resto queda por si se reimprime con otro.
DICCIONARIOS = (
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_6X6_50",
    "DICT_6X6_250",
    "DICT_APRILTAG_36h11",
)

try:  # cv2.aruco puede faltar en builds de opencv sin "contrib".
    import cv2.aruco as _aruco  # type: ignore

    ARUCO_DISPONIBLE = hasattr(_aruco, "ArucoDetector") or hasattr(_aruco, "detectMarkers")
    MOTIVO_SIN_ARUCO = "" if ARUCO_DISPONIBLE else "cv2.aruco sin ArucoDetector ni detectMarkers"
except Exception as exc:  # ImportError u otra
    _aruco = None  # type: ignore
    ARUCO_DISPONIBLE = False
    MOTIVO_SIN_ARUCO = f"cv2.aruco no disponible: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Utilidades de imagen
# ---------------------------------------------------------------------------

def leer_imagen(fuente) -> np.ndarray:
    """Acepta una ruta o un ndarray y devuelve BGR uint8 de 3 canales."""
    if isinstance(fuente, np.ndarray):
        imagen = fuente
    else:
        ruta = Path(fuente)
        if not ruta.exists():
            raise FileNotFoundError(f"No encuentro la imagen: {ruta}")
        imagen = cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)
        if imagen is None:
            raise ValueError(f"OpenCV no pudo leer la imagen: {ruta}")
    if imagen.ndim == 2:
        return cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    if imagen.shape[2] == 4:
        return cv2.cvtColor(imagen, cv2.COLOR_BGRA2BGR)
    return imagen


def ordenar_esquinas(esquinas) -> np.ndarray:
    """Devuelve TL, TR, BR, BL. Misma logica que comun.ordenar_esquinas de v1."""
    puntos = np.asarray(esquinas, dtype="float32").reshape(4, 2)
    suma = puntos.sum(axis=1)
    diferencia = np.diff(puntos, axis=1).reshape(-1)
    return np.array(
        [
            puntos[np.argmin(suma)],
            puntos[np.argmin(diferencia)],
            puntos[np.argmax(suma)],
            puntos[np.argmax(diferencia)],
        ],
        dtype="float32",
    )


def _lados_por_centroide(centros: dict[int, np.ndarray]) -> dict[int, str]:
    """Asigna TL/TR/BR/BL a cada id segun la posicion de su centro."""
    ids = list(centros)
    puntos = np.array([centros[i] for i in ids], dtype="float32")
    suma = puntos.sum(axis=1)
    diferencia = puntos[:, 0] - puntos[:, 1]
    i_tl = int(np.argmin(suma))
    i_br = int(np.argmax(suma))
    resto = [k for k in range(len(ids)) if k not in (i_tl, i_br)]
    if len(resto) != 2:
        raise ValueError("no pude separar los 4 marcadores en esquinas")
    # En coordenadas de imagen y crece hacia abajo: TR es el de mayor x-y y BL
    # el de menor.
    i_tr = resto[0] if diferencia[resto[0]] > diferencia[resto[1]] else resto[1]
    i_bl = resto[1] if i_tr == resto[0] else resto[0]
    return {
        ids[i_tl]: "TL",
        ids[i_tr]: "TR",
        ids[i_br]: "BR",
        ids[i_bl]: "BL",
    }


def _caja_area_jugable(bgr: np.ndarray) -> tuple[int, int, int, int]:
    """Caja (x0, y0, x1, y1) del tablero por el color de sus casillas.

    En la plantilla el tablero es lo unico con tono medio: las casillas azul y
    gris caen entre 70 y 235 de brillo, mientras el fondo de papel es blanco
    (255) y los marcadores/textos son negros (0). Es mas estable que buscar
    bordes porque el tablero no tiene marco propio.
    """
    canales = bgr.astype("int16")
    minimo = canales.min(axis=2)
    maximo = canales.max(axis=2)
    mascara = ((minimo > 70) & (maximo < 235)).astype("uint8")
    # Las letras en negro sobre blanco tambien dejan pixeles de antialias en el
    # rango medio. El tablero es una sola componente grande y conexa, asi que
    # quedarse con la mayor elimina esos restos sin erosionar el borde real.
    _, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    if len(estadisticas) <= 1:
        raise ValueError(
            "no encontre el area coloreada del tablero en la plantilla; "
            "revisa plantilla/tablero.png"
        )
    areas = estadisticas[1:, cv2.CC_STAT_AREA]
    etiqueta = 1 + int(np.argmax(areas))
    if int(areas.max()) < 10000:
        raise ValueError(
            "el area coloreada del tablero es demasiado pequena "
            f"({int(areas.max())} pixeles); revisa plantilla/tablero.png"
        )
    componente = etiquetas == etiqueta
    # El borde del tablero tiene 1-2 px de antialias que el umbral amplio deja
    # pasar y ensancharia la caja ~2 px por lado. Las casillas son de color
    # plano, asi que se recuperan los colores dominantes y se queda solo lo que
    # este cerca de ellos. Esa es la caja real del area jugable.
    colores, cuentas = np.unique(bgr[componente], axis=0, return_counts=True)
    if len(colores) > 1:
        dominantes = colores[np.argsort(cuentas)[-2:]].astype("int16")
        diferencias = bgr.astype("int16")[None, :, :, :] - dominantes[:, None, None, :]
        distancia = np.sqrt((diferencias.astype("float32") ** 2).sum(axis=3)).min(axis=0)
        estricta = componente & (distancia <= 8.0)
        if int(estricta.sum()) > 10000:
            componente = estricta
    ys, xs = np.nonzero(componente)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Deteccion de marcadores
# ---------------------------------------------------------------------------

def _diccionario(nombre: str):
    return _aruco.getPredefinedDictionary(getattr(_aruco, nombre))


def _detectar_con_diccionario(gris: np.ndarray, nombre: str):
    """Devuelve (dict id -> esquinas 4x2, ids crudos) para un diccionario."""
    dic = _diccionario(nombre)
    if hasattr(_aruco, "ArucoDetector"):
        detector = _aruco.ArucoDetector(dic, _aruco.DetectorParameters())
        esquinas, ids, _ = detector.detectMarkers(gris)
    else:  # API legacy de OpenCV 4.x
        detector = _aruco.DetectorParameters_create()
        esquinas, ids, _ = _aruco.detectMarkers(gris, dic, parameters=detector)
    encontrados: dict[int, np.ndarray] = {}
    if ids is None or esquinas is None:
        return encontrados
    for lados, identificador in zip(esquinas, ids.ravel()):
        entero = int(identificador)
        # Un id repetido se queda con la primera deteccion.
        if entero not in encontrados:
            encontrados[entero] = np.asarray(lados, dtype="float32").reshape(4, 2)
    return encontrados


def _detectar_marcadores(gris: np.ndarray, plantilla: "Plantilla | None" = None):
    """Devuelve (marcadores, diccionario, metodo).

    Si cv2.aruco existe se prueban los diccionarios empezando por el de la
    plantilla. Si no existe se cae al detector por plantilla (template matching).
    """
    if ARUCO_DISPONIBLE:
        orden = list(DICCIONARIOS)
        if plantilla is not None and plantilla.diccionario in orden:
            orden.remove(plantilla.diccionario)
            orden.insert(0, plantilla.diccionario)
        mejor: tuple[dict, str] | None = None
        for nombre in orden:
            try:
                encontrados = _detectar_con_diccionario(gris, nombre)
            except cv2.error:
                continue
            # Quedanos cuando ya detecta los 4: probar otros diccionarios
            # no aporta nada (4x4 vs 5x5 son marcadores distintos).
            if len(encontrados) >= 4:
                return encontrados, nombre, "aruco"
            if mejor is None or len(encontrados) > len(mejor[0]):
                mejor = (encontrados, nombre)
        if mejor is not None and mejor[0]:
            return mejor[0], mejor[1], "aruco-parcial"
        return {}, "", "aruco"

    if plantilla is None:
        raise RuntimeError(
            "cv2.aruco no esta disponible y no hay plantilla para el fallback: "
            + MOTIVO_SIN_ARUCO
        )
    return _marcadores_por_plantilla(gris, plantilla), plantilla.diccionario, "fallback-plantilla"


def _rotar_punto(u: float, v: float, alto: int, ancho: int, k: int) -> tuple[float, float]:
    """Rota (u, v) igual que ``np.rot90(..., k)`` (antihorario)."""
    k %= 4
    if k == 0:
        return u, v
    if k == 1:
        return v, ancho - 1 - u
    if k == 2:
        return ancho - 1 - u, alto - 1 - v
    return alto - 1 - v, u


def _marcadores_por_plantilla(gris: np.ndarray, plantilla: "Plantilla") -> dict[int, np.ndarray]:
    """Fallback sin cv2.aruco: template matching de cada marcador de la plantilla.

    Limitacion conocida: barre rotaciones de 90 grados, no cualquier angulo. Un
    marcador girado 45 grados puede no encontrarse. Sirve para no bloquear el
    pipeline cuando el build de OpenCV no trae el modulo aruco.
    """
    encontrados: dict[int, np.ndarray] = {}
    alto_img, ancho_img = gris.shape[:2]
    if gris.ndim != 2:
        gris = cv2.cvtColor(gris, cv2.COLOR_BGR2GRAY)

    for identificador, parche in plantilla.parches.items():
        parche = np.asarray(parche, dtype="uint8")
        alto_p, ancho_p = parche.shape[:2]
        origen_x, origen_y = plantilla.parche_origenes[identificador]
        mejor_puntaje = 1e9
        mejor: tuple[float, float, int, float, float] | None = None
        for k in range(4):
            rotado = np.rot90(parche, k)
            alto_r, ancho_r = rotado.shape[:2]
            # Escalas candidatas: el marcador en la foto puede estar entre 0.3x
            # y 2.5x del tamano que tiene en la plantilla.
            for escala in np.linspace(0.3, 2.5, 23):
                alto_s = max(8, int(round(alto_r * escala)))
                ancho_s = max(8, int(round(ancho_r * escala)))
                if alto_s >= alto_img or ancho_s >= ancho_img:
                    continue
                molde = cv2.resize(rotado, (ancho_s, alto_s), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(gris, molde, cv2.TM_CCOEFF_NORMED)
                _, maximo, _, ubicacion = cv2.minMaxLoc(res)
                if -maximo < mejor_puntaje:
                    mejor_puntaje = -maximo
                    mejor = (
                        ubicacion[0], ubicacion[1], k,
                        ancho_s / ancho_r, alto_s / alto_r,
                    )
        if mejor is None:
            continue

        x_m, y_m, k, escala_x, escala_y = mejor
        puntos = []
        for (u, v) in plantilla.esquinas_marcador(identificador):
            local_u = u - origen_x
            local_v = v - origen_y
            ru, rv = _rotar_punto(local_u, local_v, alto_p, ancho_p, k)
            puntos.append((x_m + ru * escala_x, y_m + rv * escala_y))
        encontrados[identificador] = np.array(puntos, dtype="float32").reshape(4, 2)

    return encontrados


# ---------------------------------------------------------------------------
# Plantilla
# ---------------------------------------------------------------------------

@dataclass
class Plantilla:
    """Relacion fija entre los marcadores y el area jugable, medida en la plantilla."""

    diccionario: str
    marcadores: dict[int, np.ndarray]  # id -> 4x2 en coords de la plantilla
    lado_a_id: dict[str, int]
    id_a_lado: dict[int, str]
    esquinas_area: np.ndarray  # 4x2 TL, TR, BR, BL
    caja_area: tuple[int, int, int, int]
    parches: dict[int, np.ndarray] = field(default_factory=dict)
    parche_origenes: dict[int, tuple[int, int]] = field(default_factory=dict)
    ruta: Path | None = None

    @property
    def centros(self) -> dict[int, np.ndarray]:
        return {i: c.mean(axis=0) for i, c in self.marcadores.items()}

    def ids_en_orden(self) -> list[int]:
        return [self.lado_a_id[lado] for lado in LADOS]

    def esquinas_marcador(self, identificador: int) -> np.ndarray:
        return np.asarray(self.marcadores[identificador], dtype="float32").reshape(4, 2)


def _parche_marcador(gris: np.ndarray, esquinas: np.ndarray, margen: int = 2):
    x0 = max(0, int(math.floor(esquinas[:, 0].min())) - margen)
    x1 = int(math.ceil(esquinas[:, 0].max())) + margen
    y0 = max(0, int(math.floor(esquinas[:, 1].min())) - margen)
    y1 = int(math.ceil(esquinas[:, 1].max())) + margen
    return gris[y0:y1, x0:x1].copy(), (x0, y0)


def analizar_plantilla(ruta=PLANTILLA_TABLERO) -> Plantilla:
    """Mide en la plantilla: diccionario, ids por lado y esquinas del area jugable."""
    bgr = leer_imagen(ruta)
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    marcadores, diccionario, _metodo = _detectar_marcadores(gris, None)
    if len(marcadores) != 4:
        raise RuntimeError(
            f"esperaba 4 marcadores en la plantilla {ruta} y detecte {len(marcadores)}. "
            "Revisa que la plantilla tenga los 4 ArUco visibles y sin recortar."
        )

    x0, y0, x1, y1 = _caja_area_jugable(bgr)
    # Se usan los bordes exteriores del area coloreada: el primer pixel de la
    # casilla y el ultimo, no el centro. Asi la homografia corta justo en el
    # borde del tablero y no medio pixel afuera.
    esquinas_area = np.array(
        [[x0, y0], [x1 + 1, y0], [x1 + 1, y1 + 1], [x0, y1 + 1]], dtype="float32"
    )

    lados = _lados_por_centroide({i: c.mean(axis=0) for i, c in marcadores.items()})
    lado_a_id = {lado: i for i, lado in lados.items()}
    parches: dict[int, np.ndarray] = {}
    parche_origenes: dict[int, tuple[int, int]] = {}
    for identificador, esquinas in marcadores.items():
        parche, origen = _parche_marcador(gris, esquinas)
        parches[identificador] = parche
        parche_origenes[identificador] = origen
    return Plantilla(
        diccionario=diccionario,
        marcadores=marcadores,
        lado_a_id=lado_a_id,
        id_a_lado=lados,
        esquinas_area=esquinas_area,
        caja_area=(x0, y0, x1, y1),
        parches=parches,
        parche_origenes=parche_origenes,
        ruta=Path(ruta) if not isinstance(ruta, np.ndarray) else None,
    )


# ---------------------------------------------------------------------------
# Deteccion publica
# ---------------------------------------------------------------------------

@dataclass
class Deteccion:
    """Resultado de ``detectar``."""

    esquinas: np.ndarray  # 4x2 TL, TR, BR, BL del area jugable
    ids: list[int]  # en el mismo orden TL, TR, BR, BL
    confianza: float  # 0..1
    marcadores: dict[int, np.ndarray]  # todos los marcadores vistos
    diccionario: str
    metodo: str
    rms_px: float
    mensaje: str = ""

    def como_diccionario(self) -> dict:
        return {
            "esquinas": self.esquinas.tolist(),
            "ids": list(self.ids),
            "confianza": self.confianza,
            "diccionario": self.diccionario,
            "metodo": self.metodo,
            "rms_px": self.rms_px,
            "mensaje": self.mensaje,
        }


def _emparejar(
    detectados: dict[int, np.ndarray], plantilla: Plantilla
) -> tuple[list[tuple[int, int]], str]:
    """Empareja ids detectados con ids de la plantilla.

    Primero por id exacto. Si no cuadran los 4 (otra impresion del mismo layout)
    se empareja por lado geometrico: la esquina TL de la foto con la TL de la
    plantilla, y asi. Devuelve [(id_detectado, id_plantilla)] y el metodo.
    """
    por_id = [(i, i) for i in plantilla.ids_en_orden() if i in detectados]
    if len(por_id) == 4:
        return por_id, "id"

    lados_det = _lados_por_centroide({i: c.mean(axis=0) for i, c in detectados.items()})
    if len(lados_det) != 4:
        return por_id, "parcial"

    por_lado = []
    for lado in LADOS:
        detectado = next((i for i, l in lados_det.items() if l == lado), None)
        if detectado is None:
            return por_id, "parcial"
        por_lado.append((detectado, plantilla.lado_a_id[lado]))
    return por_lado, "lado"


def detectar(fuente, plantilla: "Plantilla | None" = None) -> Deteccion:
    """Esquinas del area jugable, ids y confianza para una foto o la plantilla."""
    if plantilla is None:
        plantilla = analizar_plantilla()
    bgr = leer_imagen(fuente)
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    detectados, diccionario, metodo = _detectar_marcadores(gris, plantilla)
    if not detectados:
        return Deteccion(
            esquinas=np.zeros((4, 2), dtype="float32"),
            ids=[],
            confianza=0.0,
            marcadores={},
            diccionario=diccionario,
            metodo=metodo,
            rms_px=float("inf"),
            mensaje="no se detecto ningun marcador",
        )

    pares, modo = _emparejar(detectados, plantilla)
    if len(pares) < 4:
        # Con menos de 4 no hay homografia: se devuelve lo mejor posible y
        # confianza baja para que el llamador decida descartar la foto.
        return Deteccion(
            esquinas=np.zeros((4, 2), dtype="float32"),
            ids=[d for d, _ in pares],
            confianza=min(0.5, 0.1 * len(pares)),
            marcadores=detectados,
            diccionario=diccionario,
            metodo=metodo,
            rms_px=float("inf"),
            mensaje=f"solo {len(pares)} de 4 marcadores emparejados",
        )

    por_id_plantilla = {id_plantilla: id_detectado for id_detectado, id_plantilla in pares}
    origen_centros = np.array(
        [plantilla.centros[plantilla.lado_a_id[lado]] for lado in LADOS], dtype="float32"
    )
    destino_centros = np.array(
        [detectados[por_id_plantilla[plantilla.lado_a_id[lado]]].mean(axis=0) for lado in LADOS],
        dtype="float32",
    )
    homografia = cv2.getPerspectiveTransform(origen_centros, destino_centros)

    esquinas = cv2.perspectiveTransform(
        plantilla.esquinas_area.reshape(-1, 1, 2).astype("float32"), homografia
    ).reshape(4, 2)
    esquinas = ordenar_esquinas(esquinas)

    # Residual real de los marcadores: la homografia se ajusto con 4 centros
    # (8 grados de libertad) y cada marcador aporta 4 esquinas (16 ecuaciones).
    # Un residual grande significa que un marcador se detecto mal o que el
    # layout fisico no coincide con la plantilla.
    residuales = []
    lados_marcador = []
    # ``_emparejar`` devuelve ``(id_detectado, id_plantilla)``. Aqui estaba
    # intercambiado: se proyectaba el marcador con el id de la IMAGEN y se
    # comparaba contra el id de la PLANTILLA. Con ids iguales el error era
    # invisible, pero con el emparejado por lado daba un residual falso.
    for id_detectado, id_plantilla in pares:
        if id_plantilla not in plantilla.marcadores or id_detectado not in detectados:
            continue
        proyectadas = cv2.perspectiveTransform(
            plantilla.esquinas_marcador(id_plantilla).reshape(-1, 1, 2).astype("float32"),
            homografia,
        ).reshape(-1, 2)
        reales = detectados[id_detectado].reshape(-1, 2)
        residuales.append(np.linalg.norm(proyectadas - reales, axis=1))
        lados_marcador.append(
            float(np.mean(np.linalg.norm(reales - np.roll(reales, -1, axis=0), axis=1)))
        )
    rms = float(np.mean(np.concatenate(residuales))) if residuales else float("inf")
    lado_medio = float(np.mean([l for l in lados_marcador if l > 1e-6] or [1.0]))
    relativo = rms / max(lado_medio, 1e-6)

    cobertura = len(pares) / 4.0
    confianza = float(cobertura * math.exp(-((relativo / 0.25) ** 2)))
    confianza = max(0.0, min(1.0, confianza))

    mensaje = (
        f"{len(pares)}/4 marcadores ({modo}), rms {rms:.2f} px "
        f"({relativo * 100:.1f}% del lado)"
    )
    if modo != "id":
        mensaje += "; IDs distintos a la plantilla, emparejado por posicion"

    return Deteccion(
        esquinas=esquinas.astype("float32"),
        ids=[por_id_plantilla[plantilla.lado_a_id[lado]] for lado in LADOS],
        confianza=round(confianza, 4),
        marcadores=detectados,
        diccionario=diccionario,
        metodo=metodo,
        rms_px=round(rms, 3),
        mensaje=mensaje,
    )


# ---------------------------------------------------------------------------
# Deteccion robusta: ArUco multi-escala + cuadriculado + piezas
# ---------------------------------------------------------------------------
#
# ``detectar`` solo usa ArUco y exige los 4 marcadores. En las fotos reales eso
# deja fuera 4 de cada 5 imagenes: el marcador que falta no esta oculto, esta
# subexpuesto o es demasiado chico para el detector a resolucion nativa. Esta
# seccion encadena tres estrategias y devuelve la PRIMERA que pasa una
# validacion geometrica independiente:
#
#   1. ``aruco-multi-escala``: ArUco a varias escalas, fusionado por id, y
#      homografia con las 4 esquinas de CADA marcador (2 marcadores ya dan 8
#      correspondencias exactas; no hace falta esperar al cuarto).
#   2. ``cuadriculado``: el patron azul/gris del tablero impreso. Los centros de
#      las 32 casillas azules forman una reticula conocida; se ajusta la
#      homografia a esa reticula. No necesita ArUco.
#   3. ``piezas``: ultimo recurso, rejilla inferida de las detecciones del
#      modelo. Solo vale si la rejilla se valida.
#
# La validacion comun a las tres es ``_validar_colores``: se rectifica y se
# comprueba que las 64 casillas tengan el color que les toca (azul o gris). Una
# homografia corrida media casilla no puede acertarlas, y una girada 90 grados
# tampoco (el patron de paridad cambia). Es una prueba sobre la imagen, no sobre
# la confianza del detector.

ESCALAS_ARUCO = (1.0, 2.0, 2.5, 3.0)
LADO_VISTA = 800.0
PASO_VISTA = LADO_VISTA / 8.0
# Pixels maximos de una escala ampliada. La escala 3.0 sobre una foto de
# 4080x2296 serian 84 Mpx (~84 MB en gris y ~250 MB durante el resize); el
# limite evita quedarse sin memoria sin perder las escalas que si aportan.
MAX_PIXELES_ESCALA = 60_000_000

# Casillas AZULES de la plantilla: fila+columna impar, con la fila 0 en el lado
# de las negras (arriba en la vista rectificada canonica).
CELDAS_AZULES = [(r, c) for r in range(8) for c in range(8) if (r + c) % 2 == 1]
RETICULA_AZUL = np.array(
    [[(c + 0.5) * PASO_VISTA, (r + 0.5) * PASO_VISTA] for r, c in CELDAS_AZULES],
    dtype="float32",
)
ESQUINAS_VISTA = np.array(
    [[0.0, 0.0], [LADO_VISTA, 0.0], [LADO_VISTA, LADO_VISTA], [0.0, LADO_VISTA]],
    dtype="float32",
)
ESQUINA_DE_LADO = {"TL": (0, 0), "TR": (8, 0), "BR": (8, 8), "BL": (0, 8)}

# Umbrales de validacion. Estan medidos sobre las 134 fotos reales del dataset
# (ver docs/t4b-notas.md): una homografia correcta da >= 0.93 de casillas con el
# color esperado; una corrida media casilla cae por debajo de 0.55.
UMBRAL_COLOR = 0.80       # concordancia minima del damero impreso (-1..1)
TOLERANCIA_ARUCO = 0.05   # error de reproyeccion maximo, relativo al lado mayor

_MODELO_PIEZAS = None  # cache del modelo YOLO (cargarlo tarda segundos)


def _detectar_escala(gris: np.ndarray, diccionario: str, escala: float,
                     plantilla: "Plantilla | None" = None) -> dict[int, np.ndarray]:
    """Marcadores detectados en una escala, devueltos en coords originales."""
    alto, ancho = gris.shape[:2]
    if escala == 1.0:
        imagen = gris
    else:
        if alto * ancho * escala * escala > MAX_PIXELES_ESCALA:
            return {}
        imagen = cv2.resize(
            gris, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC
        )
    detectados = _detectar_con_diccionario(imagen, diccionario)
    if escala != 1.0:
        detectados = {i: e / escala for i, e in detectados.items()}
    return detectados


def _fusionar_marcadores(vistas: list[dict[int, np.ndarray]]) -> dict[int, np.ndarray]:
    """Promedia por id las detecciones de varias escalas.

    Pesa por el area del marcador en la imagen original: la deteccion mas grande
    tiene mas pixeles por lado y por tanto esquinas mas finas. Una deteccion
    espuria en una escala chica apenas mueve la media.
    """
    acumulado: dict[int, list[np.ndarray]] = {}
    for detectados in vistas:
        for identificador, esquinas in detectados.items():
            lados = np.linalg.norm(esquinas - np.roll(esquinas, -1, axis=0), axis=1)
            lado = float(np.mean(lados))
            if not np.isfinite(lado) or lado < 2.0:
                continue
            peso = lado * lado
            previo = acumulado.get(identificador)
            if previo is None:
                acumulado[identificador] = [esquinas * peso, np.float64(peso)]
            else:
                previo[0] = previo[0] + esquinas * peso
                previo[1] = previo[1] + peso
    return {
        i: (valores[0] / valores[1]).astype("float32")
        for i, valores in acumulado.items()
    }


def _detectar_multi_escala(gris: np.ndarray, plantilla: "Plantilla | None",
                           escalas=ESCALAS_ARUCO) -> tuple[dict[int, np.ndarray], int]:
    """ArUco PROBADO a varias escalas y fusionado. Devuelve (marcadores, escalas)."""
    if not ARUCO_DISPONIBLE:
        return {}, 0
    # Primero el diccionario de la plantilla: es el unico que puede dar ids
    # validos. Si no aparece nada, se prueban los demas como hacia ``detectar``.
    nombre = plantilla.diccionario if plantilla is not None else DICCIONARIOS[0]
    vistas = []
    for escala in escalas:
        detectados = _detectar_escala(gris, nombre, escala, plantilla)
        if detectados:
            vistas.append(detectados)
    fusionados = _fusionar_marcadores(vistas)
    if fusionados:
        return fusionados, len(vistas)

    orden = [n for n in DICCIONARIOS if n != nombre]
    for alterno in orden:
        vistas = []
        for escala in escalas:
            detectados = _detectar_escala(gris, alterno, escala, plantilla)
            if detectados:
                vistas.append(detectados)
        fusionados = _fusionar_marcadores(vistas)
        if fusionados:
            return fusionados, len(vistas)
    return {}, 0


def _homografia_marcadores(detectados: dict[int, np.ndarray], plantilla: "Plantilla"):
    """Homografia plantilla -> imagen con las esquinas de todos los marcadores.

    Devuelve (homografia | None, rms_px, numero_de_marcadores, desvio_max_px).
    Con un solo marcador las 4 correspondencias son las esquinas de un cuadrado
    chico: el ajuste existe pero extrapola muy mal, asi que el llamador solo lo
    usa como semilla. Con 2 ya hay 8 correspondencias exactas.
    """
    origen, destino, grupos = [], [], []
    for identificador, esquinas in detectados.items():
        if identificador not in plantilla.marcadores:
            continue
        reales = np.asarray(esquinas, dtype="float32").reshape(4, 2)
        if not np.isfinite(reales).all():
            continue
        plantilla_esq = plantilla.esquinas_marcador(identificador)
        origen.append(plantilla_esq)
        destino.append(reales)
        grupos.append(plantilla_esq)
    if not grupos:
        return None, float("inf"), 0, float("inf")

    origen = np.concatenate(origen).astype("float32")
    destino = np.concatenate(destino).astype("float32")
    if len(grupos) == 1:
        homografia = cv2.getPerspectiveTransform(origen, destino)
    else:
        lados = np.linalg.norm(destino - np.roll(destino, -1, axis=0), axis=1)
        lado_medio = float(np.median(lados))
        homografia, _ = cv2.findHomography(
            origen, destino, cv2.RANSAC, max(2.0, 0.25 * lado_medio)
        )
    if homografia is None or not np.isfinite(homografia).all():
        return None, float("inf"), len(grupos), float("inf")

    proyectadas = cv2.perspectiveTransform(
        origen.reshape(-1, 1, 2), homografia
    ).reshape(-1, 2)
    errores = np.linalg.norm(proyectadas - destino, axis=1)
    rms = float(np.sqrt(np.mean(errores ** 2)))
    desvio = float(errores.max())
    return homografia, rms, len(grupos), desvio


def _esquinas_desde_homografia(homografia: np.ndarray, plantilla: "Plantilla") -> np.ndarray:
    """Esquinas del area jugable en ORDEN CANONICO (no en orden de imagen).

    ``detectar`` las reordena por posicion en la imagen, lo que rectifica girada
    una foto tomada a 90 grados. Aqui se conserva el orden de la plantilla:
    TL=fila 8, TR=fila 8, BR=fila 1, BL=fila 1. Asi la vista rectificada sale
    siempre con las blancas abajo, venga como venga la camara.
    """
    return cv2.perspectiveTransform(
        plantilla.esquinas_area.reshape(-1, 1, 2).astype("float32"), homografia
    ).reshape(4, 2).astype("float32")


def _validar_colores(imagen_bgr: np.ndarray, esquinas: np.ndarray) -> float:
    """Concordancia del damero impreso con la rejilla rectificada (-1..1).

    Se rectifica y se mide el canal b* de LAB en las esquinas de cada casilla
    (el centro lo tapa la ficha). El azul tiene b* bajo y el gris b* alto, asi
    que para cada par de casillas vecinas se compara la DIFERENCIA de b* con el
    signo que impone el patron: +1 si la primera es azul, -1 si es gris.

    Es un filtro adaptado local, no un umbral global de color. Un umbral fijo de
    saturacion daba 50% en las fotos nocturnas; una k-medias global daba 70-84%
    cuando la esquina del tablero quedaba en sombra. Comparar vecinos cancela la
    iluminacion, que cambia despacio, y el patron, que cambia celda a celda.

    Valores medidos en las 134 fotos: una homografia correcta da 0.89-1.00; una
    corrida una casilla da -1.00; una corrida media casilla da ~0.00; una corrida
    un tercio de casilla (2 marcadores del mismo lado) da 0.66-0.70, y por eso el
    umbral de aceptacion es 0.80.
    """
    try:
        matriz = cv2.getPerspectiveTransform(
            np.asarray(esquinas, dtype="float32"), ESQUINAS_VISTA
        )
    except cv2.error:
        return 0.0
    vista = cv2.warpPerspective(
        imagen_bgr, matriz, (int(LADO_VISTA), int(LADO_VISTA)), flags=cv2.INTER_AREA
    )
    if vista.size == 0:
        return 0.0
    lab = cv2.cvtColor(vista, cv2.COLOR_BGR2LAB)
    margen = 0.12
    mitad = 0.05
    medidas = np.zeros(64, dtype="float64")
    # Ocho puntos por casilla: las cuatro esquinas (fuera del disco de la ficha)
    # y los cuatro medios lados. La mediana aguanta una ficha descentrada.
    puntos = (
        (margen, margen), (1 - margen, margen), (1 - margen, 1 - margen),
        (margen, 1 - margen), (0.5, margen), (0.5, 1 - margen),
        (margen, 0.5), (1 - margen, 0.5),
    )
    for fila in range(8):
        for columna in range(8):
            x0, y0 = columna * PASO_VISTA, fila * PASO_VISTA
            muestras = []
            for fx, fy in puntos:
                cx = int(round(x0 + fx * PASO_VISTA))
                cy = int(round(y0 + fy * PASO_VISTA))
                r = max(2, int(round(mitad * PASO_VISTA)))
                parche = lab[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r]
                if parche.size == 0:
                    continue
                # b* bajo = tira al azul; alto = gris/neutral.
                muestras.append(float(np.median(parche.reshape(-1, 3)[:, 2])))
            medidas[fila * 8 + columna] = (
                float(np.median(muestras)) if muestras else 128.0
            )

    total = 0.0
    peso = 0.0
    for fila in range(8):
        for columna in range(8):
            indice = fila * 8 + columna
            azul = (fila + columna) % 2 == 1
            for dfila, dcolumna in ((0, 1), (1, 0)):
                vecino_fila, vecino_columna = fila + dfila, columna + dcolumna
                if vecino_fila > 7 or vecino_columna > 7:
                    continue
                vecino = vecino_fila * 8 + vecino_columna
                diferencia = medidas[vecino] - medidas[indice]
                esperado = 1.0 if azul else -1.0
                total += esperado * diferencia
                peso += abs(diferencia)
    return float(total / peso) if peso > 1e-6 else 0.0


def _centroides_celdas(imagen_bgr: np.ndarray, max_lado: int = 1400) -> np.ndarray:
    """Centros de las casillas azules detectadas en la foto.

    Se trabaja sobre una copia reducida (nunca se modifica la original) porque
    el detector de color no necesita resolucion y las fotos de celular pesan
    9 Mpx. El tamano de casilla NO se toma de la mediana de areas: las motas de
    ruido son mayoria; se toma el grupo de areas que suma mas pixeles, que son
    las 32 casillas azules.
    """
    alto, ancho = imagen_bgr.shape[:2]
    escala = min(1.0, max_lado / max(alto, ancho))
    if escala < 1.0:
        chica = cv2.resize(imagen_bgr, None, fx=escala, fy=escala,
                           interpolation=cv2.INTER_AREA)
    else:
        chica = imagen_bgr
    hsv = cv2.cvtColor(chica, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, (85, 40, 50), (145, 255, 255))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    numero, etiquetas, estadisticas, _centros = cv2.connectedComponentsWithStats(
        mascara, connectivity=8
    )
    if numero <= 1:
        return np.zeros((0, 2), dtype="float32")
    areas = estadisticas[1:, cv2.CC_STAT_AREA].astype(float)
    anchos = estadisticas[1:, cv2.CC_STAT_WIDTH].astype(float)
    altos = estadisticas[1:, cv2.CC_STAT_HEIGHT].astype(float)
    relleno = areas / np.maximum(anchos * altos, 1.0)
    aspecto = anchos / np.maximum(altos, 1.0)
    validos = (
        (areas >= 40) & (relleno > 0.45) & (aspecto > 0.45) & (aspecto < 2.2)
        & (anchos >= 6) & (altos >= 6)
    )
    if int(validos.sum()) < 4:
        return np.zeros((0, 2), dtype="float32")

    areas_validas = areas[validos]
    registro = np.log(np.maximum(areas_validas, 1.0))
    hist, bordes = np.histogram(registro, bins=24)
    total = np.zeros(len(hist), dtype=float)
    indices = np.clip(np.digitize(registro, bordes) - 1, 0, len(hist) - 1)
    for valor, indice in zip(areas_validas, indices):
        total[indice] += valor
    grupo = int(np.argmax(total))
    centro = float(np.exp((bordes[grupo] + bordes[grupo + 1]) / 2.0))
    validos &= (areas > 0.45 * centro) & (areas < 2.4 * centro)
    if int(validos.sum()) < 4:
        return np.zeros((0, 2), dtype="float32")

    lado = float(np.sqrt(np.median(areas[validos])))
    etiquetas_validas = (np.nonzero(validos)[0] + 1).tolist()
    nucleo = np.isin(etiquetas, etiquetas_validas).astype(np.uint8) * 255
    # Dilatar media casilla y medio une el damero en un solo bloque: asi se
    # descarta el azul del fondo (cuadernos, cables) sin mirar colores raros.
    tamano = max(3, int(round(1.5 * lado)) | 1)
    bloque = cv2.dilate(nucleo, np.ones((tamano, tamano), np.uint8))
    numero_bloques, etiquetas_bloque, areas_bloque, _c = cv2.connectedComponentsWithStats(
        bloque, connectivity=8
    )
    if numero_bloques <= 1:
        return np.zeros((0, 2), dtype="float32")

    # El tablero es el bloque que contiene MAS casillas validas, no el mas
    # grande: un objeto azul del fondo puede ocupar mas pixeles que el tablero.
    cuentas = np.zeros(numero_bloques, dtype=int)
    for etiqueta in etiquetas_validas:
        ys, xs = np.nonzero(etiquetas == etiqueta)
        if xs.size == 0:
            continue
        reparto = np.bincount(etiquetas_bloque[ys, xs].ravel(), minlength=numero_bloques)
        elegido = int(np.argmax(reparto))
        if elegido > 0:
            cuentas[elegido] += 1
    if cuentas[1:].size == 0 or cuentas[1:].max() == 0:
        return np.zeros((0, 2), dtype="float32")
    tablero = 1 + int(np.argmax(cuentas[1:]))
    zona = cv2.erode(
        (etiquetas_bloque == tablero).astype(np.uint8) * 255,
        np.ones((tamano, tamano), np.uint8),
    )

    puntos = []
    for etiqueta in etiquetas_validas:
        ys, xs = np.nonzero((etiquetas == etiqueta) & (zona > 0))
        if xs.size < 20:
            continue
        puntos.append((xs.mean() / escala, ys.mean() / escala))
    return np.array(puntos, dtype="float32")


def _area_firmada(puntos: np.ndarray) -> float:
    """Area con signo del cuadrilatero; el signo dice si es giro o espejo."""
    x = puntos[:, 0]
    y = puntos[:, 1]
    return float(
        0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    )


def _candidatos_esquina(fila: int, columna: int) -> list[tuple[int, int]]:
    """Casillas azules a una celda o menos de la esquina (fila, columna)."""
    return [
        (r, c) for r, c in CELDAS_AZULES
        if abs(r - fila) <= 1 and abs(c - columna) <= 1
    ]


_CANDIDATOS: dict[tuple[int, int], list[tuple[int, int]]] = {
    (0, 0): _candidatos_esquina(0, 0),
    (0, 7): _candidatos_esquina(0, 7),
    (7, 7): _candidatos_esquina(7, 7),
    (7, 0): _candidatos_esquina(7, 0),
}
_ESQUINAS_PLACA = [(0, 0), (0, 7), (7, 7), (7, 0)]


def _punto_celda(fila: int, columna: int) -> list[float]:
    return [(columna + 0.5) * PASO_VISTA, (fila + 0.5) * PASO_VISTA]


def _puntuar_reticula(homografia: np.ndarray, centroides: np.ndarray):
    """(inliers, error_medio_px, mascara) de una homografia contra la reticula."""
    if homografia is None or not np.isfinite(homografia).all():
        return 0, float("inf"), None
    try:
        inversa = np.linalg.inv(homografia)
    except np.linalg.LinAlgError:
        return 0, float("inf"), None
    vista = cv2.perspectiveTransform(
        centroides.reshape(-1, 1, 2), inversa
    ).reshape(-1, 2)
    distancias = np.linalg.norm(vista[:, None, :] - RETICULA_AZUL[None, :, :], axis=2)
    minimas = distancias.min(axis=1)
    dentro = np.all(
        (vista > -0.3 * PASO_VISTA) & (vista < 8.3 * PASO_VISTA), axis=1
    )
    inliers = (minimas < 0.35 * PASO_VISTA) & dentro
    total = int(inliers.sum())
    error = float(minimas[inliers].mean()) if total else float("inf")
    return total, error, inliers


def _hipotesis_reticula(centroides: np.ndarray) -> list[np.ndarray]:
    """Homografias iniciales: 4 extremos -> 4 esquinas de la placa, solo giros.

    Las esquinas grises de la placa tienen dos casillas azules candidatas (no
    se sabe de que lado cae el extremo); las azules tienen una sola. Se
    descartan las asignaciones que invierten el sentido de giro: una foto de un
    tablero no puede salir espejada, solo rotada.
    """
    suma = centroides[:, 0] + centroides[:, 1]
    diferencia = centroides[:, 0] - centroides[:, 1]
    extremos = [
        int(np.argmin(suma)), int(np.argmax(diferencia)),
        int(np.argmax(suma)), int(np.argmin(diferencia)),
    ]
    if len(set(extremos)) != 4:
        return []
    hipotesis = []
    for giro in range(4):
        # girar la asignacion de esquinas equivale a rotar 90 grados la placa
        esquinas = _ESQUINAS_PLACA[giro:] + _ESQUINAS_PLACA[:giro]
        for c0 in _CANDIDATOS[esquinas[0]]:
            for c1 in _CANDIDATOS[esquinas[1]]:
                for c2 in _CANDIDATOS[esquinas[2]]:
                    for c3 in _CANDIDATOS[esquinas[3]]:
                        origen = np.array(
                            [_punto_celda(*c0), _punto_celda(*c1),
                             _punto_celda(*c2), _punto_celda(*c3)],
                            dtype="float32",
                        )
                        destino = np.array(
                            [centroides[extremos[0]], centroides[extremos[1]],
                             centroides[extremos[2]], centroides[extremos[3]]],
                            dtype="float32",
                        )
                        if _area_firmada(origen) * _area_firmada(destino) <= 0:
                            continue
                        homografia = cv2.getPerspectiveTransform(origen, destino)
                        if np.isfinite(homografia).all():
                            hipotesis.append(homografia)
    return hipotesis


def _refinar_reticula(homografia: np.ndarray, centroides: np.ndarray,
                      iteraciones: int = 6):
    """Ajuste fino por minimos cuadrados reasignando cada centro a su celda."""
    total, error, _ = _puntuar_reticula(homografia, centroides)
    mejor = (total, error, homografia)
    for _ in range(iteraciones):
        if mejor[0] < 6:
            break
        vista = cv2.perspectiveTransform(
            centroides.reshape(-1, 1, 2), np.linalg.inv(mejor[2])
        ).reshape(-1, 2)
        distancias = np.linalg.norm(
            vista[:, None, :] - RETICULA_AZUL[None, :, :], axis=2
        )
        indices = distancias.argmin(axis=1)
        _, _, inliers = _puntuar_reticula(mejor[2], centroides)
        if inliers is None:
            break
        nueva, _ = cv2.findHomography(
            RETICULA_AZUL[indices[inliers]], centroides[inliers],
            cv2.RANSAC, 0.02 * LADO_VISTA,
        )
        if nueva is None:
            break
        total_nueva, error_nueva, _ = _puntuar_reticula(nueva, centroides)
        if total_nueva < mejor[0]:
            break
        if total_nueva == mejor[0] and error_nueva >= mejor[1]:
            break
        mejor = (total_nueva, error_nueva, nueva)
    return mejor


def _homografia_por_cuadriculado(imagen_bgr: np.ndarray, semilla=None):
    """Homografia placa->imagen ajustada a la reticula azul del tablero.

    ``semilla`` es una homografia ya valida (por ejemplo la de ArUco con 2 o 3
    marcadores): sirve para no recorrer todas las hipotesis y, sobre todo, para
    conservar el sentido de giro correcto.
    """
    centroides = _centroides_celdas(imagen_bgr)
    if len(centroides) < 10:
        return None, 0, float("inf"), len(centroides)
    if semilla is not None:
        total, error, _ = _puntuar_reticula(semilla, centroides)
        # Con una semilla de un solo ArUco el ajuste puede venir corrido: se
        # refina igual y la validacion de color decide. Exigir 10 inliers aqui
        # descartaba semillas que el refinamiento si rescataba.
        if total >= 6:
            total, error, homografia = _refinar_reticula(semilla, centroides)
            if total >= 10:
                return homografia, total, error, len(centroides)
    candidatas = _hipotesis_reticula(centroides)
    if not candidatas:
        return None, 0, float("inf"), len(centroides)
    puntuadas = []
    for homografia in candidatas:
        total, error, _ = _puntuar_reticula(homografia, centroides)
        if total >= 10:
            puntuadas.append((total, error, homografia))
    puntuadas.sort(key=lambda t: (-t[0], t[1]))
    for _total, _error, homografia in puntuadas[:5]:
        total, error, refinada = _refinar_reticula(homografia, centroides)
        if total >= 10:
            return refinada, total, error, len(centroides)
    return None, 0, float("inf"), len(centroides)


def _esquinas_por_cuadricula_sb(imagen_bgr: np.ndarray):
    """Intento con ``cv2.findChessboardCornersSB`` sobre la mascara de color.

    El damero azul/gris se convierte en un tablero blanco y negro, que es lo que
    espera OpenCV. Un tablero de 8x8 casillas tiene 7x7 cruces interiores, asi
    que ese es el patron que corresponde; se prueba tambien (9,9) por si la hoja
    trae un borde extra que el detector prefiera. Devuelve (homografia, patron).
    """
    damero, escala = _mascara_damero(imagen_bgr)
    if damero is None:
        return None, ""
    for filas, columnas in ((7, 7), (9, 9)):
        try:
            encontrado, esquinas = cv2.findChessboardCornersSB(
                damero, (filas, columnas),
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
            )
        except cv2.error:
            continue
        if not encontrado or esquinas is None or len(esquinas) != filas * columnas:
            continue
        puntos = esquinas.reshape(-1, 2) / escala
        # Los cruces interiores de un tablero de (filas+1)x(columnas+1) casillas
        # caen en las fracciones (i+1)/(columnas+1) del area jugable.
        origen = np.array(
            [[(columna + 1) * LADO_VISTA / (columnas + 1),
              (fila + 1) * LADO_VISTA / (filas + 1)]
             for fila in range(filas) for columna in range(columnas)],
            dtype="float32",
        )
        homografia, _ = cv2.findHomography(origen, puntos.astype("float32"),
                                           cv2.RANSAC, 3.0)
        if homografia is not None and np.isfinite(homografia).all():
            return homografia, f"sb-{filas}x{columnas}"
    return None, ""


def _mascara_damero(imagen_bgr: np.ndarray, max_lado: int = 1000):
    """Damero binario (azul=blanco) recortado al bloque del tablero."""
    alto, ancho = imagen_bgr.shape[:2]
    escala = min(1.0, max_lado / max(alto, ancho))
    chica = (
        cv2.resize(imagen_bgr, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
        if escala < 1.0 else imagen_bgr
    )
    hsv = cv2.cvtColor(chica, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, (85, 40, 50), (145, 255, 255))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    tamano = max(5, int(min(mascara.shape) * 0.03) | 1)
    bloque = cv2.dilate(mascara, np.ones((tamano, tamano), np.uint8))
    contornos, _ = cv2.findContours(bloque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None, escala
    contorno = max(contornos, key=cv2.contourArea)
    relleno = np.zeros_like(mascara)
    cv2.drawContours(relleno, [contorno], -1, 255, -1)
    relleno = cv2.erode(relleno, np.ones((tamano, tamano), np.uint8))
    limpia = cv2.bitwise_and(mascara, relleno)
    if (limpia > 0).sum() < 0.05 * limpia.size:
        return None, escala
    return limpia, escala


def _orientar(esquinas: np.ndarray, marcadores: dict[int, np.ndarray],
              imagen_bgr: np.ndarray, plantilla: "Plantilla") -> tuple[np.ndarray, str]:
    """Decide entre la vista canonica y la girada 180 grados.

    El damero no distingue esas dos: girar 180 grados deja las casillas azules en
    casillas azules. Los ids de los marcadores si: el 0 esta en el lado de las
    negras a la izquierda, el 3 en el de las blancas a la izquierda. Sin
    marcadores se usa la oscuridad de las fichas (las negras van arriba en la
    posicion inicial). Si nada decide, se deja la de la imagen y se avisa.
    """
    girada = np.array(
        [esquinas[2], esquinas[3], esquinas[0], esquinas[1]], dtype="float32"
    )
    candidatas = [(esquinas, "imagen"), (girada, "imagen")]
    ids_por_lado = {identificador: lado for lado, identificador in plantilla.lado_a_id.items()}

    if marcadores:
        puntajes = []
        for candidata, _ in candidatas:
            matriz = cv2.getPerspectiveTransform(
                candidata.astype("float32"), ESQUINAS_VISTA
            )
            aciertos = 0
            for identificador, esquinas_marcador in marcadores.items():
                lado = ids_por_lado.get(identificador)
                if lado is None:
                    continue
                centro = np.asarray(esquinas_marcador, dtype="float32").reshape(4, 2).mean(axis=0)
                punto = cv2.perspectiveTransform(
                    centro.reshape(-1, 1, 2).astype("float32"), matriz
                ).reshape(2)
                esperado = np.array(ESQUINA_DE_LADO[lado], dtype="float32")
                if (punto[0] - 4 * PASO_VISTA) * (esperado[0] - 4 * PASO_VISTA) >= 0 and \
                   (punto[1] - 4 * PASO_VISTA) * (esperado[1] - 4 * PASO_VISTA) >= 0:
                    aciertos += 1
            puntajes.append(aciertos)
        if puntajes[0] != puntajes[1]:
            return (candidatas[0][0], "marcadores") if puntajes[0] > puntajes[1] \
                else (candidatas[1][0], "marcadores")

    def oscuridad(candidata: np.ndarray) -> float:
        matriz = cv2.getPerspectiveTransform(candidata.astype("float32"), ESQUINAS_VISTA)
        vista = cv2.warpPerspective(
            imagen_bgr, matriz, (int(LADO_VISTA), int(LADO_VISTA)),
            flags=cv2.INTER_AREA,
        )
        gris = cv2.cvtColor(vista, cv2.COLOR_BGR2GRAY)
        oscuro = (gris < 90).mean(axis=1)
        return float(oscuro[50:290].mean() - oscuro[510:750].mean())

    arriba, abajo = oscuridad(candidatas[0][0]), oscuridad(candidatas[1][0])
    if abs(arriba - abajo) > 0.05:
        return (candidatas[0][0], "fichas") if arriba > abajo else (candidatas[1][0], "fichas")
    return candidatas[0][0], "supuesta"


def _esquinas_por_piezas(imagen_bgr: np.ndarray, plantilla: "Plantilla"):
    """Ultimo recurso: rejilla inferida de las detecciones del modelo.

    Se reutiliza ``comun.esquinas_por_rejilla``, que ya valida la rejilla
    (piezas dentro del tablero, en casillas distintas y centradas). El modelo es
    el ``best.pt`` de v1, entrenado con las fichas viejas: por eso es el ultimo
    recurso y ademas se exige la validacion de color.
    """
    global _MODELO_PIEZAS
    try:
        from ultralytics import YOLO
        from comun import esquinas_por_rejilla, normalizar_detecciones
    except Exception:
        return None, "sin modelo"
    modelo_path = Path(r"F:\TAVO\Projects\ajedrez-yolo\models\best.pt")
    if not modelo_path.exists():
        return None, f"no encuentro {modelo_path}"
    if _MODELO_PIEZAS is None:
        _MODELO_PIEZAS = YOLO(str(modelo_path))
    resultados = _MODELO_PIEZAS.predict(
        imagen_bgr, conf=0.25, agnostic_nms=True, verbose=False
    )
    if not resultados:
        return None, "sin predicciones"
    cajas = resultados[0].boxes
    if cajas is None or len(cajas) == 0:
        return None, "sin cajas"
    nombres = resultados[0].names
    detecciones = []
    for caja in cajas:
        clase = nombres[int(caja.cls[0])]
        x0, y0, x1, y1 = (float(v) for v in caja.xyxy[0].tolist())
        detecciones.append((clase, (x0 + x1) / 2.0, (y0 + y1) / 2.0,
                            float(caja.conf[0])))
    esquinas, motivo = esquinas_por_rejilla(normalizar_detecciones(detecciones))
    if esquinas is None:
        return None, motivo
    return np.asarray(esquinas, dtype="float32").reshape(4, 2), motivo


def detectar_robusto(imagen, plantilla: "Plantilla | None" = None,
                     usar_piezas: bool = True):
    """Esquinas del area jugable encadenando estrategias.

    Devuelve ``(esquinas, metodo, detalle)``:

    * ``esquinas``: 4x2 en orden CANONICO (TL, TR, BR, BL de la placa; las
      blancas abajo), o ``None`` si ninguna estrategia paso la validacion.
    * ``metodo``: ``"aruco-multi-escala"``, ``"cuadriculado"``, ``"piezas"`` o
      ``"ninguno"``.
    * ``detalle``: dict con ``ids``, ``escalas``, ``marcadores``, ``color``,
      ``orientacion``, ``rms_px``, ``motivo`` y ``cuadricula_inliers`` para el
      CSV de auditoria.

    La validacion es la misma para todas: la rejilla rectificada tiene que
    concordar con el damero impreso del tablero al menos ``UMBRAL_COLOR``
    (ver ``_validar_colores``), que es una prueba sobre la imagen y no sobre la
    confianza del detector.
    """
    if plantilla is None:
        plantilla = analizar_plantilla()
    bgr = leer_imagen(imagen)
    alto, ancho = bgr.shape[:2]
    lado_mayor = float(max(alto, ancho))
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    detalle: dict = {
        "ids": [], "escalas": 0, "marcadores": 0, "color": 0.0,
        "orientacion": "", "rms_px": None, "motivo": "", "cuadricula_inliers": 0,
    }

    # --- 1. ArUco multi-escala ------------------------------------------------
    marcadores, escalas = _detectar_multi_escala(gris, plantilla)
    detalle["ids"] = sorted(marcadores)
    detalle["marcadores"] = len(marcadores)
    detalle["escalas"] = escalas
    semilla = None
    if marcadores:
        homografia, rms, cuantos, desvio = _homografia_marcadores(marcadores, plantilla)
        if homografia is not None:
            esquinas = _esquinas_desde_homografia(homografia, plantilla)
            color = _validar_colores(bgr, esquinas)
            detalle.update(color=round(color, 4), rms_px=round(rms, 3))
            # Con 2 o mas marcadores la homografia es una respuesta valida. Con
            # uno solo extrapola demasiado: sirve de semilla, no de resultado.
            if (cuantos >= 2 and color >= UMBRAL_COLOR
                    and desvio <= TOLERANCIA_ARUCO * lado_mayor):
                detalle["orientacion"] = "marcadores"
                detalle["motivo"] = (
                    f"{cuantos} marcadores, rms {rms:.2f} px, "
                    f"color {color:.2f}"
                )
                return esquinas, "aruco-multi-escala", detalle
            detalle["motivo"] = (
                f"aruco {cuantos} marcadores descartado: color {color:.2f}, "
                f"desvio {desvio:.0f} px"
            )
            semilla = homografia
    else:
        detalle["motivo"] = f"{len(marcadores)} marcadores"

    # --- 2. Cuadriculado del tablero -----------------------------------------
    homografia, inliers, error, cuantos_centros = _homografia_por_cuadriculado(
        bgr, semilla
    )
    if homografia is None:
        homografia, patron = _esquinas_por_cuadricula_sb(bgr)
        if homografia is not None:
            inliers, error = 0, 0.0
            detalle["motivo"] = f"OpenCV {patron}"
    detalle["cuadricula_inliers"] = int(inliers)
    if homografia is not None:
        # La homografia de la reticula mapea coordenadas de placa (0..800) a la
        # imagen; las esquinas de la placa son las mismas que las de la vista.
        esquinas = cv2.perspectiveTransform(
            ESQUINAS_VISTA.reshape(-1, 1, 2).astype("float32"), homografia
        ).reshape(4, 2).astype("float32")
        color = _validar_colores(bgr, esquinas)
        if color < UMBRAL_COLOR:
            homografia, color = _mejor_desplazamiento(bgr, homografia, color)
            esquinas = cv2.perspectiveTransform(
                ESQUINAS_VISTA.reshape(-1, 1, 2).astype("float32"), homografia
            ).reshape(4, 2).astype("float32")
        if color >= UMBRAL_COLOR:
            esquinas, orientacion = _orientar(esquinas, marcadores, bgr, plantilla)
            color = _validar_colores(bgr, esquinas)
            detalle.update(
                color=round(color, 4), orientacion=orientacion,
                motivo=(f"reticula {inliers}/{cuantos_centros} casillas, "
                        f"error {error:.1f}/800, color {color:.2f}, "
                        f"orientacion {orientacion}"),
            )
            return esquinas, "cuadriculado", detalle
        detalle["motivo"] += f"; cuadricula descartada: color {color:.2f}"

    # --- 3. Piezas ------------------------------------------------------------
    if usar_piezas:
        esquinas, motivo = _esquinas_por_piezas(bgr, plantilla)
        if esquinas is not None:
            color = _validar_colores(bgr, esquinas)
            if color >= UMBRAL_COLOR:
                esquinas, orientacion = _orientar(esquinas, marcadores, bgr, plantilla)
                color = _validar_colores(bgr, esquinas)
                detalle.update(
                    color=round(color, 4), orientacion=orientacion,
                    motivo=f"rejilla de piezas ({motivo}), color {color:.2f}",
                )
                return esquinas, "piezas", detalle
            detalle["motivo"] += f"; piezas descartadas: color {color:.2f}"
        else:
            detalle["motivo"] += f"; piezas: {motivo}"

    detalle["motivo"] = detalle["motivo"] or "sin estrategia valida"
    return None, "ninguno", detalle


def _mejor_desplazamiento(imagen_bgr: np.ndarray, homografia: np.ndarray,
                          actual: float):
    """Prueba desplazar la rejilla una casilla y se queda con lo mejor.

    El damero tiene una ambiguedad de una casilla: el mismo patron se repite
    cada dos. Cuando una deteccion (tipicamente ``findChessboardCornersSB``
    sobre la mascara) cae una casilla corrida, la concordancia da -1.00 y la
    foto se perderia aunque la geometria sea perfecta. Se prueban los 8 vecinos
    y solo se aceptan si superan a la original.
    """
    mejor_color = actual
    mejor_homografia = homografia
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            traslacion = np.array(
                [[1.0, 0.0, dx * PASO_VISTA], [0.0, 1.0, dy * PASO_VISTA], [0.0, 0.0, 1.0]]
            )
            candidata = homografia @ traslacion
            esquinas = cv2.perspectiveTransform(
                ESQUINAS_VISTA.reshape(-1, 1, 2).astype("float32"), candidata
            ).reshape(4, 2).astype("float32")
            color = _validar_colores(imagen_bgr, esquinas)
            if color > mejor_color:
                mejor_color = color
                mejor_homografia = candidata
    return mejor_homografia, mejor_color


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

def dibujar_debug(fuente, deteccion: "Deteccion | None" = None,
                  plantilla: "Plantilla | None" = None) -> np.ndarray:
    """Devuelve la imagen anotada con esquinas, lados, IDs y texto de estado."""
    imagen = leer_imagen(fuente).copy()
    if plantilla is None:
        plantilla = analizar_plantilla()
    if deteccion is None:
        deteccion = detectar(fuente, plantilla)

    alto, ancho = imagen.shape[:2]
    grosor = max(1, int(round(min(alto, ancho) / 400)))
    fuente_texto = cv2.FONT_HERSHEY_SIMPLEX
    escala_texto = max(0.35, min(alto, ancho) / 1100.0)

    # Esquinas del area jugable.
    if deteccion.esquinas.any():
        poligono = deteccion.esquinas.astype("int32").reshape(-1, 1, 2)
        cv2.polylines(imagen, [poligono], True, (0, 220, 0), grosor * 2, cv2.LINE_AA)
        for (x, y), lado in zip(deteccion.esquinas, LADOS):
            color = (0, 220, 0)
            cv2.circle(imagen, (int(x), int(y)), grosor * 3, color, -1, cv2.LINE_AA)
            cv2.putText(
                imagen, lado, (int(x) + 6, int(y) - 6), fuente_texto, escala_texto,
                color, grosor, cv2.LINE_AA,
            )
        # Etiquetas de los lados por su punto medio.
        medios = [
            (deteccion.esquinas[0] + deteccion.esquinas[1]) / 2,
            (deteccion.esquinas[1] + deteccion.esquinas[2]) / 2,
            (deteccion.esquinas[2] + deteccion.esquinas[3]) / 2,
            (deteccion.esquinas[3] + deteccion.esquinas[0]) / 2,
        ]
        for (x, y), nombre in zip(medios, ("N", "E", "S", "O")):
            cv2.putText(
                imagen, nombre, (int(x) - 4, int(y) - 4), fuente_texto, escala_texto,
                (255, 160, 0), grosor, cv2.LINE_AA,
            )

    # Marcadores: poligono + id.
    for identificador, esquinas in deteccion.marcadores.items():
        poligono = np.asarray(esquinas, dtype="int32").reshape(-1, 1, 2)
        cv2.polylines(imagen, [poligono], True, (0, 160, 255), grosor, cv2.LINE_AA)
        centro = np.asarray(esquinas).reshape(4, 2).mean(axis=0)
        lado = deteccion.ids.index(identificador) if identificador in deteccion.ids else None
        etiqueta = f"id={identificador}"
        if lado is not None:
            etiqueta += f" {LADOS[lado]}"
        cv2.putText(
            imagen, etiqueta,
            (int(centro[0]) + 4, int(centro[1]) + 4), fuente_texto, escala_texto * 0.9,
            (0, 160, 255), grosor, cv2.LINE_AA,
        )

    lineas = [
        f"aruco: {deteccion.diccionario or 'ninguno'} ({deteccion.metodo})",
        f"ids: {deteccion.ids}",
        f"confianza: {deteccion.confianza:.2f}  rms: {deteccion.rms_px:.2f} px",
        deteccion.mensaje,
    ]
    for indice, linea in enumerate(lineas):
        if not linea:
            continue
        y = int((indice + 1) * 18 * max(1.0, escala_texto * 2.2))
        cv2.putText(
            imagen, linea, (8, y), fuente_texto, escala_texto * 1.15, (30, 30, 30),
            grosor * 3, cv2.LINE_AA,
        )
        cv2.putText(
            imagen, linea, (8, y), fuente_texto, escala_texto * 1.15, (255, 255, 255),
            grosor, cv2.LINE_AA,
        )
    return imagen


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _imprimir(resultado: Deteccion, ruta) -> None:
    print(f"imagen: {ruta}")
    print(f"diccionario: {resultado.diccionario or 'ninguno'}  metodo: {resultado.metodo}")
    print(f"ids detectados (TL,TR,BR,BL): {resultado.ids}")
    print(f"marcadores vistos: {sorted(resultado.marcadores)}")
    print(f"confianza: {resultado.confianza:.3f}  rms: {resultado.rms_px:.2f} px")
    print(f"detalle: {resultado.mensaje}")
    if resultado.esquinas.any():
        for lado, (x, y) in zip(LADOS, resultado.esquinas):
            print(f"  esquina {lado}: ({x:.1f}, {y:.1f})")
    else:
        print("  sin esquinas: no hay homografia posible")


def main(argv=None) -> int:
    analizador = argparse.ArgumentParser(
        description="Detecta las 4 esquinas del area jugable por marcadores ArUco."
    )
    analizador.add_argument(
        "--plantilla", default=str(PLANTILLA_TABLERO),
        help="plantilla del tablero con los 4 ArUco (default: plantilla/tablero.png)",
    )
    analizador.add_argument(
        "--imagen", default=None,
        help="imagen a analizar; si se omite se analiza la propia plantilla",
    )
    analizador.add_argument(
        "--debug-salida", default=None,
        help="ruta donde guardar la vista anotada (opcional)",
    )
    analizador.add_argument(
        "--fallback", action="store_true",
        help="forzar el detector por template matching aunque cv2.aruco exista",
    )
    analizador.add_argument(
        "--robusto", action="store_true",
        help="usar detectar_robusto (ArUco multi-escala + cuadriculado + piezas)",
    )
    analizador.add_argument(
        "--sin-piezas", action="store_true",
        help="con --robusto, no usar el modelo YOLO como ultimo recurso",
    )
    argumentos = analizador.parse_args(argv)

    plantilla = analizar_plantilla(argumentos.plantilla)
    print("plantilla:", argumentos.plantilla)
    print(f"  diccionario: {plantilla.diccionario}")
    print(f"  ids por lado: {plantilla.lado_a_id}")
    x0, y0, x1, y1 = plantilla.caja_area
    print(f"  caja inclusiva de pixeles: x {x0}..{x1}, y {y0}..{y1}")
    print(f"  caja del area jugable: {plantilla.caja_area}")
    print("  esquinas del area jugable (TL,TR,BR,BL):")
    for lado, (x, y) in zip(LADOS, plantilla.esquinas_area):
        print(f"    {lado}: ({x:.1f}, {y:.1f})")

    ruta = argumentos.imagen or argumentos.plantilla
    if argumentos.robusto:
        esquinas, metodo, detalle = detectar_robusto(
            ruta, plantilla, usar_piezas=not argumentos.sin_piezas
        )
        print()
        print(f"imagen: {ruta}")
        print(f"metodo: {metodo}")
        print(f"ids: {detalle.get('ids')}  marcadores: {detalle.get('marcadores')}  "
              f"escalas: {detalle.get('escalas')}")
        print(f"color: {detalle.get('color')}  orientacion: {detalle.get('orientacion')}  "
              f"cuadricula: {detalle.get('cuadricula_inliers')}")
        print(f"detalle: {detalle.get('motivo')}")
        if esquinas is not None:
            for lado, (x, y) in zip(LADOS, esquinas):
                print(f"  esquina {lado}: ({x:.1f}, {y:.1f})")
        else:
            print("  sin esquinas: ninguna estrategia paso la validacion")
        if argumentos.debug_salida and esquinas is not None:
            anotada = leer_imagen(ruta).copy()
            cv2.polylines(anotada, [esquinas.astype("int32").reshape(-1, 1, 2)],
                          True, (0, 220, 0), 6, cv2.LINE_AA)
            cv2.putText(anotada, f"{metodo}: {detalle.get('motivo', '')}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(anotada, f"{metodo}: {detalle.get('motivo', '')}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            destino = Path(argumentos.debug_salida)
            destino.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destino), anotada)
            print(f"vista anotada: {destino}")
        return 0

    if argumentos.fallback:
        global ARUCO_DISPONIBLE
        estado = ARUCO_DISPONIBLE
        ARUCO_DISPONIBLE = False
        try:
            resultado = detectar(ruta, plantilla)
        finally:
            ARUCO_DISPONIBLE = estado
    else:
        resultado = detectar(ruta, plantilla)

    print()
    _imprimir(resultado, ruta)

    if argumentos.debug_salida:
        anotada = dibujar_debug(ruta, resultado, plantilla)
        destino = Path(argumentos.debug_salida)
        destino.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destino), anotada)
        print(f"vista anotada: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
