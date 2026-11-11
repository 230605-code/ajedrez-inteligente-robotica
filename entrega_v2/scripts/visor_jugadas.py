"""Visor de JUGADAS en vivo para el juego real con la ESP32-CAM.

Es la interfaz funcional de la sesion: mantiene el ESTADO de la partida en
``assets/previews/jugadas/estado.json`` y, en cada paso, deduce QUE jugada se
hizo comparando la lectura fisica del tablero contra ese estado, la valida con
``python-chess``, pide la mejor respuesta a Stockfish y deja una imagen con la
jugada detectada y la sugerencia.

Modos (uno por corrida):

    --iniciar                 lee el tablero con la camara (3 capturas, voto de
                              mayoria por casilla, anti-fantasmas + legalizar)
                              y guarda el ESTADO BASE.
    --iniciar-fen "<FEN>"     ESTADO BASE desde una FEN declarada: la FEN manda
                              (no toca la camara). El turno sale del campo
                              activo de la FEN ('w' blanca / 'b' negra).
    --paso                    UN paso de partida: captura 2-3 lecturas, vota por
                              casilla, deduce la jugada, la aplica, pide el
                              mejor movimiento y guarda paso_NN/.
    --estado                  imprime el estado actual SIN tocar la camara.
    --descargar-stockfish     baja el binario oficial a tools/stockfish/.

Reusa TODO lo medido: ``pipeline_fen`` (clasificador, FEN, legalidad,
fantasmas), ``conector_juego.LectorFenV2`` (captura + fusion YOLO + filtro) y
``esquinas_aruco`` (rectificacion). Este modulo no reimplementa vision.

Salidas (todo dentro de assets/previews/jugadas/):
    estado.json                       estado de la partida (lo lee --estado)
    paso_NN/cenital.jpg               tablero 800x800 + pie verde + motor naranja
    paso_NN/paso.txt                  FEN antes/despues, jugada UCI+SAN, motor
    paso_NN/crudo.jpg                 el frame de la camara que se uso
    paso_NN/FALLO.txt + falla_NN.jpg  cuando la deduccion NO cierra (no actualiza)

Uso:
    python scripts/visor_jugadas.py --iniciar --ip 10.251.192.200
    python scripts/visor_jugadas.py --paso
    python scripts/visor_jugadas.py --estado

Es headless a proposito: nunca abre una ventana (no hay cv2.imshow).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

import chess  # noqa: E402

SALIDA = RAIZ / "assets" / "previews" / "jugadas"
ESTADO = SALIDA / "estado.json"
DESTINO_CENITAL = np.array([[0, 0], [799, 0], [799, 799], [0, 799]],
                           dtype="float32")
DESTINO_STOCKFISH = RAIZ / "tools" / "stockfish" / "stockfish.exe"
IP_POR_DEFECTO = "10.251.192.200"

LADO = 800
CASILLA_PX = LADO // 8
ALTO_PIE = 170
VERDE = (60, 200, 60)
NARANJA = (0, 150, 255)
BLANCO = (235, 235, 235)
GRIS = (170, 170, 170)

CASILLAS = [chess.square_name(c) for c in chess.SQUARES]

PIEZA_A_NOMBRE = {
    chess.PAWN: "peon", chess.ROOK: "torre", chess.KNIGHT: "caballo",
    chess.BISHOP: "alfil", chess.QUEEN: "dama", chess.KING: "rey",
}
NOMBRE_A_PIEZA = {v: k for k, v in PIEZA_A_NOMBRE.items()}
PROMOCION = {"dama": chess.QUEEN, "torre": chess.ROOK, "alfil": chess.BISHOP,
             "caballo": chess.KNIGHT}
NOMBRE_PROMOCION = {v: k for k, v in PROMOCION.items()}

URLS_STOCKFISH = (
    ("sf_17.1", "stockfish-windows-x86-64-avx2"),
    ("sf_17.1", "stockfish-windows-x86-64-sse41-popcnt"),
    ("sf_16", "stockfish-windows-x86-64-avx2"),
    ("sf_16", "stockfish-windows-x86-64-sse41-popcnt"),
)


# ---------------------------------------------------------------------------
# Casillas y estado
# ---------------------------------------------------------------------------

def _clase_de_pieza(pieza: chess.Piece | None) -> str:
    """Clase de 13 del clasificador ('peon_b', 'vacia'...) para una pieza."""
    if pieza is None:
        return "vacia"
    return f"{PIEZA_A_NOMBRE[pieza.piece_type]}_{'b' if pieza.color else 'n'}"


def _ocupacion_de_tablero(tablero: chess.Board) -> dict[str, str]:
    """{casilla: clase} con las 64 casillas del tablero."""
    return {casilla: _clase_de_pieza(tablero.piece_at(chess.parse_square(casilla)))
            for casilla in CASILLAS}


def _color_de_clase(clase: str) -> str:
    return clase.rsplit("_", 1)[-1] if "_" in clase else ""


def _es_peon(clase: str) -> bool:
    return clase.startswith("peon_")


def _tablero_ascii(tablero: chess.Board) -> str:
    """Tablero legible sin depender de la fuente unicode de la consola."""
    lineas = []
    for fila in range(8):
        rango = 8 - fila
        celdas = []
        for columna in range(8):
            pieza = tablero.piece_at(chess.square(columna, rango - 1))
            celdas.append(pieza.symbol() if pieza else ".")
        lineas.append(f" {rango} | " + " ".join(celdas))
    lineas.append("   +" + "-" * 16)
    lineas.append("     " + " ".join("abcdefgh"))
    return "\n".join(lineas)


def _derechos_enroque(tablero: chess.Board) -> None:
    """Reconstruye los derechos de enroque desde la POSICION de las piezas.

    Una posicion leida por la camara llega sin derechos: ``construir_fen`` arma
    un ``chess.Board(None)`` y el FEN sale con '-'. Sin derechos, ``legal_moves``
    nunca ofrece el enroque y una jugada real de enroque se reportaria como
    ilegal. Se conceden solo si rey y torre estan en su casilla original y las
    casillas intermedias estan vacias (lo mismo que exige la regla).
    """
    derechos = []
    for color, primera, letras in ((chess.WHITE, 0, "KQ"),
                                   (chess.BLACK, 7, "kq")):
        rey = tablero.piece_at(chess.square(4, primera))
        if not (rey and rey.piece_type == chess.KING and rey.color == color):
            continue
        torre_h = tablero.piece_at(chess.square(7, primera))
        if torre_h and torre_h.piece_type == chess.ROOK and torre_h.color == color:
            if not any(tablero.piece_at(chess.square(f, primera))
                       for f in (5, 6)):
                derechos.append(letras[0])
        torre_a = tablero.piece_at(chess.square(0, primera))
        if torre_a and torre_a.piece_type == chess.ROOK and torre_a.color == color:
            if not any(tablero.piece_at(chess.square(f, primera))
                       for f in (1, 2, 3)):
                derechos.append(letras[1])
    tablero.set_castling_fen("".join(derechos) or "-")


def _cargar_estado() -> dict:
    if not ESTADO.exists():
        raise SystemExit(
            "no hay estado de partida. Empieza con --iniciar (camara) o\n"
            '  --iniciar-fen "<FEN>" (posicion declarada).')
    return json.loads(ESTADO.read_text(encoding="utf-8"))


def _guardar_estado(estado: dict) -> None:
    ESTADO.parent.mkdir(parents=True, exist_ok=True)
    ESTADO.write_text(json.dumps(estado, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")


def _tablero_de_estado(estado: dict) -> chess.Board:
    try:
        tablero = chess.Board(estado["fen"])
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"el estado guardado no tiene una FEN valida: {exc}")
    esperado = chess.WHITE if estado.get("turno") == "blanca" else chess.BLACK
    if tablero.turn != esperado:
        tablero.turn = esperado
    return tablero


# ---------------------------------------------------------------------------
# Deduccion de la jugada
# ---------------------------------------------------------------------------

@dataclass
class Deducion:
    estado: str                       # "jugada" | "ninguna" | "fallo"
    move: chess.Move | None = None
    origen: str = ""
    destino: str = ""
    captura: bool = False
    enroque: bool = False
    al_paso: bool = False
    promocion: bool = False
    san: str = ""
    uci: str = ""
    nota: str = ""
    motivo: str = ""
    movidas: list[tuple[str, str]] = field(default_factory=list)


def _pares_por_clase(origenes: list[str], destinos: list[str],
                     base: dict[str, str], actual: dict[str, str]
                     ) -> list[tuple[str, str]] | None:
    """Empareja origenes y destinos que muestran EXACTAMENTE la misma clase."""
    libres = list(destinos)
    pares = []
    for origen in origenes:
        clase = base[origen]
        candidato = next((d for d in libres if actual[d] == clase), None)
        if candidato is None:
            return None
        libres.remove(candidato)
        pares.append((origen, candidato))
    return pares if not libres else None


def deducir_jugada(base: dict[str, str], actual: dict[str, str],
                   tablero: chess.Board, promocion: str = "dama") -> Deducion:
    """Compara la ocupacion base con la leida y deduce la jugada.

    ``base`` es la foto del estado y ``actual`` la lectura nueva (ambas
    {casilla: clase}). ``tablero`` es el ``chess.Board`` del estado: con EL se
    decide si la jugada deducida es legal y se genera el SAN.

    Reglas:
      * 1 origen + 1 destino con la misma clase -> jugada normal (o captura si
        el destino tenia una pieza en el estado base).
      * 2 origenes + 2 destinos con clases matching -> ENROQUE (rey + torre).
      * 2 origenes + 1 destino con un peon de cada color -> captura al paso.
      * peon al destino en la ultima fila -> promocion (se usa la pieza que
        muestra la lectura, o ``promocion`` con una NOTA).
      * cualquier otra combinacion -> ``fallo`` (no se actualiza el estado).
    """
    origenes = sorted(c for c in CASILLAS
                      if base.get(c, "vacia") != "vacia"
                      and actual.get(c, "vacia") == "vacia")
    destinos = sorted(c for c in CASILLAS
                      if actual.get(c, "vacia") != "vacia"
                      and actual.get(c) != base.get(c))

    if not origenes and not destinos:
        return Deducion(estado="ninguna",
                        motivo="la lectura es igual al estado base")
    if not origenes or not destinos:
        return Deducion(
            estado="fallo", movidas=list(zip(origenes, destinos)),
            motivo=(f"desbalance: {len(origenes)} casillas vaciadas "
                    f"{origenes} y {len(destinos)} ocupadas {destinos}"))

    if len(origenes) == 1 and len(destinos) == 1:
        return _cerrar(tablero, origenes[0], destinos[0], base, actual,
                       promocion)

    if len(origenes) == 2 and len(destinos) == 2:
        pares = _pares_por_clase(origenes, destinos, base, actual)
        if pares:
            reyes = [(o, d) for o, d in pares if base[o].startswith("rey_")]
            if len(reyes) == 1:
                deduccion = _cerrar(tablero, reyes[0][0], reyes[0][1], base,
                                    actual, promocion)
                if deduccion.estado == "jugada":
                    deduccion.enroque = True
                return deduccion
        return Deducion(
            estado="fallo", movidas=list(zip(origenes, destinos)),
            motivo=("dos piezas movidas y no forman un enroque "
                    "(no hay un rey entre los origenes o las clases no "
                    "emparejan)"))

    if len(origenes) == 2 and len(destinos) == 1:
        destino = destinos[0]
        clase_llega = actual[destino]
        mueven = [o for o in origenes if base[o] == clase_llega]
        if len(mueven) == 1 and _es_peon(clase_llega):
            capturado = [o for o in origenes if o != mueven[0]][0]
            if (_es_peon(base[capturado])
                    and _color_de_clase(base[capturado])
                    != _color_de_clase(clase_llega)):
                deduccion = _cerrar(tablero, mueven[0], destino, base, actual,
                                    promocion)
                if deduccion.estado == "jugada":
                    deduccion.al_paso = True
                    deduccion.captura = True
                    deduccion.nota = (deduccion.nota + " ").strip() + \
                        " captura al paso: se acredita el peon capturado"
                return deduccion
        return Deducion(
            estado="fallo", movidas=list(zip(origenes, destinos * 2)),
            motivo=(f"se vaciaron {origenes} y solo aparece {destino} "
                    f"({clase_llega}); no es captura al paso"))

    return Deducion(
        estado="fallo", movidas=list(zip(origenes, destinos)),
        motivo=(f"se movieron {len(origenes)} piezas ({origenes}) hacia "
                f"{destinos}: mas de una jugada en un solo paso"))


def _cerrar(tablero: chess.Board, origen: str, destino: str,
            base: dict[str, str], actual: dict[str, str],
            promocion: str) -> Deducion:
    """Valida el par origen/destino y devuelve la Deducion correspondiente."""
    clase_origen = base.get(origen, "vacia")
    clase_destino = actual.get(destino, "vacia")
    nota = ""
    pieza_promocion = None

    en_ultima = chess.square_rank(chess.parse_square(destino)) in (0, 7)
    if _es_peon(clase_origen) and en_ultima:
        if (clase_destino != clase_origen
                and _color_de_clase(clase_destino) == _color_de_clase(clase_origen)
                and clase_destino.rsplit("_", 1)[0] in NOMBRE_A_PIEZA):
            pieza_promocion = NOMBRE_A_PIEZA[clase_destino.rsplit("_", 1)[0]]
        if pieza_promocion is None:
            pieza_promocion = PROMOCION[promocion]
            nota = (f"peon en la ultima fila ({destino}): asumo "
                    f"{promocion}/{_color_de_clase(clase_origen)}; declara la "
                    "pieza real con --promocion")
    elif clase_destino != clase_origen:
        return Deducion(
            estado="fallo", movidas=[(origen, destino)],
            motivo=(f"{destino} muestra '{clase_destino}' pero {origen} tenia "
                    f"'{clase_origen}': clase o color distinto en el destino"))

    move = chess.Move(chess.parse_square(origen), chess.parse_square(destino),
                      promotion=pieza_promocion)
    deduccion = Deducion(
        estado="jugada", move=move, origen=origen, destino=destino,
        captura=base.get(destino, "vacia") != "vacia",
        promocion=pieza_promocion is not None, uci=move.uci(), nota=nota)
    if move in tablero.legal_moves:
        deduccion.san = tablero.san(move)
    else:
        deduccion.san = move.uci()
        deduccion.motivo = "ilegal"
    return deduccion


def _aplicar(tablero: chess.Board, move: chess.Move,
             al_paso: bool = False) -> chess.Board | None:
    """Board con la jugada aplicada, o None si no es legal.

    La captura al paso necesita ``ep_square`` en la FEN, que el estado no
    conserva; cuando se dedujo al paso se prueba con la casilla de paso puesta.
    """
    if move in tablero.legal_moves:
        nuevo = tablero.copy()
        nuevo.push(move)
        return nuevo
    if al_paso:
        copia = tablero.copy()
        copia.ep_square = move.to_square - 8 if tablero.turn == chess.WHITE \
            else move.to_square + 8
        if move in copia.legal_moves:
            copia.push(move)
            return copia
    return None


# ---------------------------------------------------------------------------
# Lectura fisica (reusa LectorFenV2: captura + fusion YOLO + legalidad)
# ---------------------------------------------------------------------------

def _leer(args, carpeta: Path | None = None):
    """``--lecturas`` capturas, voto de mayoria por casilla.

    Devuelve ``(ocupacion, discrepancias, confianzas, frame, cuantas)``.
    El frame es el de la lectura MAS confiable (el que se dibuja).
    """
    from conector_juego import LectorFenV2

    fijo = None
    if getattr(args, "frame", None):
        fijo = cv2.imread(str(args.frame), cv2.IMREAD_COLOR)
        if fijo is None:
            raise SystemExit(f"no pude leer el frame {args.frame}")

    lector = LectorFenV2(ip=args.ip,
                         carpeta_salida=carpeta or (SALIDA / "_lecturas"))
    capturados: list[np.ndarray] = []
    original = lector.capturar

    def capturar(intentos: int = 3):
        frame = fijo.copy() if fijo is not None else original(intentos=intentos)
        capturados.append(frame)
        return frame

    lector.capturar = capturar  # replay offline: misma ruta de codigo

    lecturas = max(1, min(5, args.lecturas))
    datos = []
    for indice in range(lecturas):
        try:
            tablero, fen, detalle = lector.leer_tablero(
                guardar=False, pausa=0.6 if indice else 0.0)
        except RuntimeError as exc:
            if indice == 0:
                raise
            print(f"  aviso: lectura {indice + 1}/{lecturas} fallo ({exc}); "
                  f"sigo con {len(datos)}")
            break
        datos.append((tablero, fen, detalle))

    ocupaciones = [_ocupacion_de_tablero(t) for t, _, _ in datos]
    confianzas = [float(d.get("confianza_media") or 0.0) for _, _, d in datos]
    mejor = int(np.argmax(confianzas))
    votos_por_casilla = {c: [o[c] for o in ocupaciones] for c in CASILLAS}

    ocupacion: dict[str, str] = {}
    discrepancias: list[str] = []
    for casilla in CASILLAS:
        votos = votos_por_casilla[casilla]
        conteo = {clase: votos.count(clase) for clase in set(votos)}
        tope = max(conteo.values())
        ganadores = [clase for clase, n in conteo.items() if n == tope]
        elegido = ganadores[0] if len(ganadores) == 1 else votos[mejor]
        if len(conteo) > 1:
            discrepancias.append(casilla)
        ocupacion[casilla] = elegido

    frame = capturados[min(mejor, len(capturados) - 1)] if capturados else None
    return ocupacion, discrepancias, confianzas, frame, len(datos)


def _cenital_de(frame: np.ndarray):
    """Rectifica el frame a 800x800 (reusa la deteccion robusta de esquinas)."""
    from esquinas_aruco import detectar_robusto

    esquinas, metodo, detalle = detectar_robusto(frame)
    if esquinas is None:
        return None, f"sin esquinas ({detalle.get('motivo') or metodo})"
    matriz = cv2.getPerspectiveTransform(
        np.asarray(esquinas, dtype="float32"), DESTINO_CENITAL)
    cenital = cv2.warpPerspective(frame, matriz, (LADO, LADO),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))
    return cenital, metodo


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------

def _centro(casilla: str) -> tuple[int, int]:
    columna = ord(casilla[0]) - ord("a")
    fila = 8 - int(casilla[1])
    return columna * CASILLA_PX + CASILLA_PX // 2, fila * CASILLA_PX + CASILLA_PX // 2


def _flecha(imagen, origen: str, destino: str, color) -> None:
    x0, y0 = _centro(origen)
    x1, y1 = _centro(destino)
    dx, dy = x1 - x0, y1 - y0
    cola = (int(x0 + dx * 0.18), int(y0 + dy * 0.18))
    punta = (int(x0 + dx * 0.88), int(y0 + dy * 0.88))
    cv2.arrowedLine(imagen, cola, punta, color, 9, cv2.LINE_AA, tipLength=0.22)
    cv2.circle(imagen, (x0, y0), 13, color, 3, cv2.LINE_AA)


def _texto(imagen, texto: str, xy: tuple[int, int], escala: float, color,
           grosor: int = 1) -> None:
    """Texto que se achica hasta entrar en el ancho util."""
    escala = max(0.32, escala)
    while escala > 0.32:
        ancho, _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala,
                                   grosor)[0]
        if ancho <= imagen.shape[1] - 2 * xy[0]:
            break
        escala -= 0.03
    cv2.putText(imagen, texto, xy, cv2.FONT_HERSHEY_SIMPLEX, escala, color,
                grosor, cv2.LINE_AA)


def _segmentos(tablero: chess.Board, move: chess.Move
               ) -> list[tuple[str, str]]:
    """(origen, destino) a dibujar; el enroque suma la torre."""
    segmentos = [(chess.square_name(move.from_square),
                  chess.square_name(move.to_square))]
    if tablero.is_castling(move):
        primera = chess.square_rank(move.from_square)
        corto = move.to_square > move.from_square
        torre_desde = chess.square(7 if corto else 0, primera)
        torre_hasta = chess.square(5 if corto else 3, primera)
        segmentos.append((chess.square_name(torre_desde),
                          chess.square_name(torre_hasta)))
    return segmentos


def _anotar(cenital, segmentos, san, segmentos_mejor, san_mejor, fen, turno,
            extra: str = "") -> np.ndarray:
    """Tablero + flechas + pie de texto con la jugada, el motor y la FEN."""
    lienzo = np.zeros((LADO + ALTO_PIE, LADO, 3), dtype=np.uint8)
    lienzo[:LADO] = cenital

    for origen, destino in segmentos_mejor:
        _flecha(lienzo, origen, destino, NARANJA)
    for origen, destino in segmentos:
        _flecha(lienzo, origen, destino, VERDE)
        for casilla in (origen, destino):
            x, y = _centro(casilla)
            cv2.rectangle(lienzo, (x - 46, y - 46), (x + 46, y + 46), VERDE, 3)

    _texto(lienzo, f"Jugada: {san}", (14, LADO + 42), 1.0, VERDE, 2)
    _texto(lienzo, f"Mejor: {san_mejor or '(sin motor)'}", (14, LADO + 88),
           0.9, NARANJA, 2)
    _texto(lienzo, f"FEN: {fen}", (14, LADO + 126), 0.55, BLANCO, 1)
    _texto(lienzo, f"turno: {turno}   {extra}".strip(), (14, LADO + 156),
           0.55, GRIS, 1)
    return lienzo


# ---------------------------------------------------------------------------
# Stockfish
# ---------------------------------------------------------------------------

def _buscar_stockfish() -> Path | None:
    """Ruta del binario. Reusa la busqueda del conector (no se duplica)."""
    from conector_juego import _ruta_stockfish

    return _ruta_stockfish()


def _probar_binario(ruta: Path) -> tuple[bool, str]:
    import subprocess

    try:
        proceso = subprocess.run([str(ruta)], input="uci\nquit\n",
                                 capture_output=True, text=True, timeout=40,
                                 cwd=str(ruta.parent))
    except Exception as exc:  # noqa: BLE001 - se reporta el motivo real
        return False, f"{type(exc).__name__}: {exc}"
    salida = (proceso.stdout or "") + (proceso.stderr or "")
    primera = salida.strip().splitlines()[0] if salida.strip() else "sin salida"
    return "uciok" in salida, primera


def descargar_stockfish(destino: Path = DESTINO_STOCKFISH,
                        verboso: bool = True) -> tuple[Path | None, str]:
    """Baja el binario oficial de GitHub releases y lo deja como stockfish.exe.

    Sin curl ni PowerShell: ``urllib`` + ``zipfile``. Prueba AVX2 y, si el
    binario no arranca, sse41-popcnt; despues las mismas dos de la version 16.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    for etiqueta, nombre in URLS_STOCKFISH:
        url = ("https://github.com/official-stockfish/Stockfish/releases/"
               f"download/{etiqueta}/{nombre}.zip")
        temporal = destino.parent / f"{nombre}.zip"
        if verboso:
            print(f"  probando {etiqueta}/{nombre}.zip")
        try:
            peticion = urllib.request.Request(
                url, headers={"User-Agent": "ajedrez-v2-visor/1.0"})
            with urllib.request.urlopen(peticion, timeout=120) as respuesta, \
                    open(temporal, "wb") as salida:
                total = int(respuesta.headers.get("Content-Length") or 0)
                recibido = 0
                while True:
                    bloque = respuesta.read(262144)
                    if not bloque:
                        break
                    salida.write(bloque)
                    recibido += len(bloque)
                    if verboso and total:
                        print(f"\r    {recibido / 1e6:6.1f}/{total / 1e6:.1f} MB",
                              end="", flush=True)
                if verboso:
                    print()
            with zipfile.ZipFile(temporal) as paquete:
                ejecutables = [n for n in paquete.namelist()
                               if n.lower().endswith(".exe")]
                if not ejecutables:
                    raise RuntimeError("el zip no trae ningun .exe")
                miembro = max(ejecutables,
                              key=lambda n: paquete.getinfo(n).file_size)
                with paquete.open(miembro) as origen, \
                        open(destino, "wb") as salida:
                    shutil.copyfileobj(origen, salida)
        except Exception as exc:  # noqa: BLE001 - se prueba el siguiente
            print(f"    fallo {nombre}: {type(exc).__name__}: {str(exc)[:90]}")
            continue
        bien, detalle = _probar_binario(destino)
        if bien:
            return destino, url
        print(f"    {nombre} no respondio 'uci' ({detalle}); pruebo otra build")
    return None, ""


def _mejores_jugadas(tablero: chess.Board, ruta: Path, tiempo_ms: int,
                     cuantas: int = 3) -> list[dict]:
    """Las ``cuantas`` mejores jugadas del motor con SAN, score y linea."""
    import chess.engine

    with chess.engine.SimpleEngine.popen_uci(str(ruta)) as motor:
        infos = motor.analyse(tablero, chess.engine.Limit(time=tiempo_ms / 1000.0),
                              multipv=cuantas)
    if isinstance(infos, dict):
        infos = [infos]
    lineas = []
    for indice, info in enumerate(sorted(infos,
                                         key=lambda i: i.get("multipv", 0))):
        pv = info.get("pv") or []
        if not pv:
            continue
        move = pv[0]
        try:
            san = tablero.san(move)
        except ValueError:
            san = move.uci()
        score = info.get("score")
        if score is None:
            texto = ""
        else:
            relativo = score.pov(tablero.turn)
            texto = f"#{relativo.mate()}" if relativo.is_mate() else \
                f"{relativo.score() / 100:+.2f}"
        lineas.append({
            "orden": indice + 1, "uci": move.uci(), "san": san,
            "score": texto,
            "linea": [m.uci() for m in pv[:4]],
        })
    return lineas


def _resolver_motor(args, tablero: chess.Board):
    """Devuelve (lineas, ruta, aviso). Nunca rompe el paso por el motor."""
    if getattr(args, "sin_motor", False):
        return [], None, "motor desactivado con --sin-motor"
    ruta = _buscar_stockfish()
    aviso = ""
    if ruta is None and not getattr(args, "sin_descarga", False):
        print("Stockfish no esta instalado: descargando la version oficial...")
        ruta, _ = descargar_stockfish()
    if ruta is None:
        return [], None, ("sin Stockfish: no hay sugerencia del motor "
                          "(descargalo con --descargar-stockfish)")
    try:
        lineas = _mejores_jugadas(tablero, ruta, args.tiempo_ms)
    except Exception as exc:  # noqa: BLE001 - el paso sigue sin motor
        return [], ruta, f"el motor fallo ({type(exc).__name__}: {exc}); sigue sin sugerencia"
    if not lineas:
        return [], ruta, "el motor no devolvio jugadas"
    return lineas, ruta, aviso


# ---------------------------------------------------------------------------
# Modos
# ---------------------------------------------------------------------------

def cmd_iniciar(args) -> int:
    ocupacion, discrepancias, confianzas, frame, cuantas = _leer(args)
    from pipeline_fen import construir_fen

    fen = construir_fen(ocupacion, turno_blanca=(args.turno == "blanca"))
    tablero = chess.Board(fen)
    _derechos_enroque(tablero)
    estado_legal = tablero.status()
    if estado_legal:  # Status.VALID es 0 (falsy): se compara por verdad
        print(f"AVISO: la posicion leida no es legal ({estado_legal.name}); "
              "revisa luces/fichas antes de jugar")

    estado = {
        "fen": tablero.fen(),
        "turno": "blanca" if tablero.turn == chess.WHITE else "negra",
        "ocupacion": _ocupacion_de_tablero(tablero),
        "historial": [],
        "contador_pasos": 0,
    }
    _guardar_estado(estado)

    carpeta_base = SALIDA / "base"
    carpeta_base.mkdir(parents=True, exist_ok=True)
    metodo = ""
    if frame is not None:
        cv2.imwrite(str(carpeta_base / "crudo.jpg"), frame)
        cenital, metodo = _cenital_de(frame)
        if cenital is not None:
            cv2.imwrite(str(carpeta_base / "cenital.jpg"), cenital)
    if estado_legal:
        (carpeta_base / "AVISO.txt").write_text(
            "el estado base NO es confiable.\n"
            f"estado del tablero: {estado_legal.name}\n"
            f"FEN leida: {tablero.fen()}\n"
            f"casillas que discrepan entre lecturas: "
            f"{', '.join(discrepancias) or '-'}\n"
            "revisa que el tablero quepa COMPLETO con margen (4 esquinas "
            "ArUco visibles), la luz y que las fichas esten en su casilla; "
            "despues vuelve a correr --iniciar.\n",
            encoding="utf-8")

    fichas = sum(1 for clase in ocupacion.values() if clase != "vacia")
    print(f"lecturas: {cuantas}   confianzas: "
          f"{', '.join(f'{c:.3f}' for c in confianzas)}")
    print(f"casillas con discrepancia entre lecturas: "
          f"{len(discrepancias)} {discrepancias if discrepancias else ''}")
    print(f"fichas: {fichas}")
    if metodo:
        print(f"esquinas: {metodo}")
    print(_tablero_ascii(tablero))
    print(f"FEN: {tablero.fen()}")
    print(f"turno: {estado['turno']}"
          + ("  (el lector asume blancas; usa --turno negra o --iniciar-fen "
             "si juegan negras)" if args.turno == "blanca" else ""))

    # A partir de la posicion detectada: la PRIMERA sugerencia de Stockfish
    # (así --iniciar ya decide la jugada que sigue, no solo guarda el estado).
    sugeridas: list[dict] = []
    aviso_motor = ""
    if not estado_legal:
        sugeridas, ruta_motor, aviso_motor = _resolver_motor(args, tablero)
        if aviso_motor:
            print(f"AVISO motor: {aviso_motor}")

    cenital, _ = _cenital_de(frame)
    if cenital is not None:
        if sugeridas:
            mejor = sugeridas[0]
            san_mejor = mejor["san"]
            move_sugerido = chess.Move.from_uci(mejor["uci"])
            segmentos_m = _segmentos(tablero, move_sugerido)
        else:
            san_mejor, segmentos_m = "", []
        anotada = _anotar(cenital, [], "(posicion inicial: mueve blancas)",
                          segmentos_m, san_mejor, tablero.fen(),
                          estado["turno"],
                          extra=(f"score mejor: {sugeridas[0]['score']}"
                                 if sugeridas else aviso_motor))
        ruta_anotada = carpeta_base / "sugerencia.jpg"
        cv2.imwrite(str(ruta_anotada), anotada)
        print(f"conejo de jugadas: {ruta_anotada.name} "
              "(flecha naranja = la jugada sugerida)")
        if sugeridas:
            print(f"sugerencia del motor: {mejor['uci']}  SAN {san_mejor}  "
                  f"({mejor['score']})")
            for linea in sugeridas:
                print(f"  {linea['orden']}. {linea['san']} ({linea['score']})")
        print(f"guardado: {ruta_anotada}")
    return 0


def cmd_iniciar_fen(args) -> int:
    try:
        tablero = chess.Board(args.iniciar_fen.strip())
    except ValueError as exc:
        print(f"ERROR: FEN invalida: {exc}")
        return 1
    estado = {
        "fen": tablero.fen(),
        "turno": "blanca" if tablero.turn == chess.WHITE else "negra",
        "ocupacion": _ocupacion_de_tablero(tablero),
        "historial": [],
        "contador_pasos": 0,
    }
    if args.turno != estado["turno"]:
        print(f"AVISO: la FEN declara turno '{estado['turno']}'; manda la FEN")
    _guardar_estado(estado)
    print("estado base desde FEN declarada (la FEN manda)")
    print(_tablero_ascii(tablero))
    print(f"FEN: {tablero.fen()}")
    print(f"turno: {estado['turno']}")
    print(f"estado base: {ESTADO}")
    return 0


def cmd_paso(args) -> int:
    estado = _cargar_estado()
    base = {c: estado["ocupacion"].get(c, "vacia") for c in CASILLAS}
    tablero = _tablero_de_estado(estado)
    numero = int(estado.get("contador_pasos", 0)) + 1
    carpeta = SALIDA / f"paso_{numero:02d}"

    ocupacion, discrepancias, confianzas, frame, cuantas = _leer(args)
    deduccion = deducir_jugada(base, ocupacion, tablero, args.promocion)

    print(f"paso {numero:02d}   lecturas: {cuantas}   "
          f"confianzas: {', '.join(f'{c:.3f}' for c in confianzas)}")
    print(f"casillas con discrepancia entre lecturas: {len(discrepancias)}")

    if deduccion.estado == "ninguna":
        print("jugada: ninguna detectada")
        print(f"  el tablero leido coincide con el estado base "
              f"({sum(1 for c in base.values() if c != 'vacia')} fichas)")
        return 0

    carpeta.mkdir(parents=True, exist_ok=True)
    _limpiar_intento(carpeta, dejar_fallo=(deduccion.estado == "fallo"))
    if frame is not None:
        cv2.imwrite(str(carpeta / "crudo.jpg"), frame)
    cenital, metodo = _cenital_de(frame) if frame is not None else (None, "")

    if deduccion.estado == "fallo":
        return _reportar_fallo(carpeta, numero, deduccion, cenital, tablero,
                               estado, ocupacion)

    print(f"  jugada deducida: {deduccion.uci}  SAN {deduccion.san}"
          + ("  (captura)" if deduccion.captura else "")
          + ("  (enroque)" if deduccion.enroque else "")
          + ("  (promocion)" if deduccion.promocion else "")
          + ("  (al paso)" if deduccion.al_paso else ""))
    if deduccion.nota:
        print(f"  NOTA: {deduccion.nota}")

    nuevo = _aplicar(tablero, deduccion.move, deduccion.al_paso)
    aviso_legal = ""
    if nuevo is None:
        aviso_legal = ("jugada detectada fuera del legales del estado pasado "
                       "(posible deteccion floja); valide con el usuario")
        print(f"  AVISO: {aviso_legal}")
        print(f"  jugada ilegal en el estado: {deduccion.uci} "
              f"({tablero.fen()})")
        nuevo = _tablero_de_lectura(ocupacion, tablero)

    lineas, ruta_motor, aviso_motor = _resolver_motor(args, nuevo)
    if aviso_motor:
        print(f"  AVISO motor: {aviso_motor}")
    mejor = lineas[0] if lineas else None
    san_mejor = mejor["san"] if mejor else ""
    if mejor:
        print(f"  mejor del motor: {mejor['uci']}  SAN {san_mejor}  "
              f"({mejor['score']})")
        print("  mejores: " + " | ".join(
            f"{l['orden']}) {l['san']} {l['score']}" for l in lineas))

    segmentos = _segmentos(tablero, deduccion.move)
    segmentos_mejor = _segmentos(nuevo, _uci_a_move(mejor["uci"])) if mejor else []
    turno_nuevo = "blanca" if nuevo.turn == chess.WHITE else "negra"
    lienzo = _anotar(cenital if cenital is not None else np.zeros((LADO, LADO, 3), np.uint8),
                     segmentos, deduccion.san, segmentos_mejor, san_mejor,
                     nuevo.fen(), turno_nuevo,
                     extra=f"lecturas {cuantas} | {metodo}")
    cv2.imwrite(str(carpeta / "cenital.jpg"), lienzo)

    (carpeta / "paso.txt").write_text(
        f"paso {numero:02d}\n"
        f"FEN antes       : {tablero.fen()}\n"
        f"turno antes     : {estado.get('turno', '')}\n"
        f"jugada deducida : {deduccion.uci}  SAN {deduccion.san}\n"
        f"captura         : {'si' if deduccion.captura else 'no'}\n"
        f"enroque         : {'si' if deduccion.enroque else 'no'}\n"
        f"promocion       : {'si' if deduccion.promocion else 'no'}\n"
        f"al paso         : {'si' if deduccion.al_paso else 'no'}\n"
        f"nota            : {deduccion.nota or '-'}\n"
        f"aviso           : {aviso_legal or '-'}\n"
        f"FEN despues     : {nuevo.fen()}\n"
        f"best move       : "
        f"{(mejor['uci'] + '  SAN ' + mejor['san']) if mejor else '(sin motor)'}\n"
        + ("mejores         :\n"
           + "\n".join(f"  {l['orden']}) {l['uci']} SAN {l['san']} {l['score']}"
                       for l in lineas) + "\n" if lineas else "")
        + f"lecturas        : {cuantas}\n"
        f"confianzas      : {', '.join(f'{c:.3f}' for c in confianzas)}\n"
        f"discrepancias   : {', '.join(discrepancias) or '-'}\n"
        f"fuente          : {args.frame or ('camara ' + args.ip)}\n"
        f"esquinas        : {metodo or '-'}\n"
        f"stockfish       : {ruta_motor or '(no disponible)'}\n",
        encoding="utf-8")

    estado["fen"] = nuevo.fen()
    estado["turno"] = turno_nuevo
    estado["ocupacion"] = _ocupacion_de_tablero(nuevo)
    estado["historial"].append(deduccion.san)
    estado["contador_pasos"] = numero
    _guardar_estado(estado)

    print(f"  FEN: {nuevo.fen()}")
    print(f"  guardado en {carpeta}")
    print(f"  historial: {' '.join(estado['historial'])}")
    return 0


def _limpiar_intento(carpeta: Path, dejar_fallo: bool) -> None:
    """Quita los artefactos del intento ANTERIOR en la misma carpeta.

    Un FALLO no avanza el contador, así que reintentar el mismo paso reutiliza
    ``paso_NN``; sin esta limpieza quedarían juntos el ``FALLO.txt`` viejo y el
    ``paso.txt`` nuevo. Solo se tocan archivos que escribe este módulo.
    """
    sobran = (("paso.txt", "cenital.jpg") if dejar_fallo
              else ("FALLO.txt",))
    for nombre in sobran:
        ruta = carpeta / nombre
        if ruta.exists():
            ruta.unlink()
    if not dejar_fallo:
        for vieja in carpeta.glob("falla_*.jpg"):
            vieja.unlink()


def _tablero_de_lectura(ocupacion: dict[str, str],
                        tablero_base: chess.Board) -> chess.Board:
    """Board con la ocupacion NUEVA re-legalizada (sin pasar por la jugada)."""
    from pipeline_fen import construir_fen

    fen = construir_fen(ocupacion,
                        turno_blanca=(tablero_base.turn == chess.BLACK))
    nuevo = chess.Board(fen)
    nuevo.turn = not tablero_base.turn
    _derechos_enroque(nuevo)
    estado_legal = nuevo.status()
    if estado_legal:  # Status.VALID es 0 (falsy)
        print(f"  AVISO: la ocupacion nueva tampoco es legal "
              f"({estado_legal.name})")
    return nuevo


def _uci_a_move(uci: str) -> chess.Move:
    return chess.Move.from_uci(uci)


def _reportar_fallo(carpeta: Path, numero: int, deduccion: Deducion,
                    cenital, tablero: chess.Board, estado: dict,
                    ocupacion: dict[str, str]) -> int:
    """Escribe FALLO.txt + falla_NN.jpg. NO toca el estado ni llama al motor."""
    casillas = []
    for origen, destino in deduccion.movidas:
        casillas.extend([origen, destino])
    casillas = list(dict.fromkeys(c for c in casillas if isinstance(c, str)
                                  and len(c) == 2))

    lienzo = (cenital.copy() if cenital is not None
              else np.zeros((LADO, LADO, 3), dtype=np.uint8))
    for casilla in casillas:
        try:
            x, y = _centro(casilla)
        except (IndexError, ValueError):
            continue
        cv2.rectangle(lienzo, (x - 46, y - 46), (x + 46, y + 46),
                      (40, 60, 240), 3)
        _texto(lienzo, casilla, (x - 26, y + 8), 0.7, (40, 60, 240), 2)
    lienzo = _anotar(lienzo, [], "(sin jugada)", [], "", tablero.fen(),
                     estado.get("turno", ""), extra="FALLO de deduccion")
    cv2.imwrite(str(carpeta / f"falla_{numero:02d}.jpg"), lienzo)

    detalle = []
    for casilla in sorted(set(casillas)):
        base = estado["ocupacion"].get(casilla, "vacia")
        ahora = ocupacion.get(casilla, "vacia")
        if base != ahora:
            detalle.append(f"  {casilla}: base '{base}' -> leido '{ahora}'")
    (carpeta / "FALLO.txt").write_text(
        f"paso {numero:02d}: la deduccion NO cierra\n"
        f"motivo           : {deduccion.motivo}\n"
        f"casillas posibles: {', '.join(casillas) or '-'}\n"
        f"movidas          : "
        f"{', '.join(f'{o}->{d}' for o, d in deduccion.movidas) or '-'}\n"
        f"FEN del estado   : {tablero.fen()}\n"
        f"ocurrencias      : el estado NO se actualizo y el motor NO se corrio\n"
        + ("\ncambios por casilla:\n" + "\n".join(detalle) + "\n"
           if detalle else ""),
        encoding="utf-8")

    print(f"  FALLO: {deduccion.motivo}")
    print(f"  casillas posibles: {', '.join(casillas) or '-'}")
    print(f"  estado SIN cambios (contador sigue en "
          f"{estado.get('contador_pasos', 0)}); sin motor")
    print(f"  detalle en {carpeta / 'FALLO.txt'}")
    return 2


def cmd_estado(args) -> int:
    estado = _cargar_estado()
    tablero = _tablero_de_estado(estado)
    historial = estado.get("historial") or []
    print(json.dumps(estado, ensure_ascii=False, indent=2))
    print()
    print(_tablero_ascii(tablero))
    print(f"FEN: {tablero.fen()}")
    print(f"turno: {estado.get('turno')}")
    print(f"pasos: {estado.get('contador_pasos', 0)}")
    print(f"historial SAN: {' '.join(historial) if historial else '(vacio)'}")
    return 0


def cmd_descargar(args) -> int:
    if DESTINO_STOCKFISH.exists():
        bien, detalle = _probar_binario(DESTINO_STOCKFISH)
        if bien:
            print(f"Stockfish ya esta en {DESTINO_STOCKFISH} "
                  f"({DESTINO_STOCKFISH.stat().st_size / 1e6:.1f} MB, responde "
                  f"'uci': {detalle})")
            return 0
        print(f"el binario existente no responde: {detalle}; se re-descarga")
    ruta, url = descargar_stockfish()
    if ruta is None:
        print("ERROR: no se pudo descargar Stockfish; el visor seguira "
              "funcionando SIN sugerencias del motor")
        return 1
    bien, detalle = _probar_binario(ruta)
    print(f"Stockfish en {ruta} ({ruta.stat().st_size / 1e6:.1f} MB)")
    print(f"origen: {url}")
    print(f"prueba 'uci': {'OK' if bien else 'FALLO'} ({detalle})")
    return 0 if bien else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modos = parser.add_mutually_exclusive_group(required=True)
    modos.add_argument("--iniciar", action="store_true",
                       help="ESTADO BASE leyendo el tablero con la camara")
    modos.add_argument("--iniciar-fen", metavar="FEN",
                       help='ESTADO BASE desde una FEN declarada: "r1bq... w - - 0 10"')
    modos.add_argument("--paso", action="store_true",
                       help="un paso de partida (captura, deduce, motor)")
    modos.add_argument("--estado", action="store_true",
                       help="imprime el estado actual sin tocar la camara")
    modos.add_argument("--descargar-stockfish", action="store_true",
                       help="baja Stockfish a tools/stockfish/stockfish.exe")

    parser.add_argument("--ip", default=IP_POR_DEFECTO,
                        help=f"IP de la ESP32-CAM (defecto {IP_POR_DEFECTO})")
    parser.add_argument("--turno", choices=("blanca", "negra"), default="blanca",
                        help="turno del estado base al iniciar con camara")
    parser.add_argument("--frame", metavar="RUTA",
                        help="usa esta imagen como frame de camara (replay "
                             "offline; mismo codigo de vision)")
    parser.add_argument("--lecturas", type=int, default=3,
                        help="capturas por paso para el voto de mayoria (2-3)")
    parser.add_argument("--promocion", choices=tuple(PROMOCION), default="dama",
                        help="pieza que se asume en una promocion no declarada")
    parser.add_argument("--tiempo-ms", type=int, default=500,
                        help="tiempo por jugada del motor (ms)")
    parser.add_argument("--sin-motor", action="store_true",
                        help="no consulta Stockfish en este paso")
    parser.add_argument("--sin-descarga", action="store_true",
                        help="no intenta bajar Stockfish automaticamente")
    args = parser.parse_args(argv)

    if args.descargar_stockfish:
        return cmd_descargar(args)
    if args.estado:
        return cmd_estado(args)
    if args.iniciar_fen:
        return cmd_iniciar_fen(args)
    if args.iniciar:
        try:
            return cmd_iniciar(args)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return 1
    try:
        return cmd_paso(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
