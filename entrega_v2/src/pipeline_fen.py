"""Pipeline v2: recorte rectificado 800x800 -> FEN, con filtro de legalidad.

Es el corazon de la arquitectura v2 (NeA armarlo desde v1: el tablero ya esta
rectificado y conocemos sus 64 casillas; no necesitamos deteccion para saber
donde mirar).

Pasos:
  1. Clasificador por casilla (MobileNetV3-Small, 13 clases = 12 piezas + vacia)
     decide que hay en cada casilla, y guarda sus ALTERNATIVAS (probabilidades).
  2. `python-chess` hace el post-filtro de movimientos legales: un tablero que
     viola las reglas (2 reyes del mismo bando, 9 peones, peon en primera fila,
     reyes adyacentes...) se corrige sustituyendo la ficha culpable por su
     2a opcion. Es correccion GRATIS: no entrenamiento adicional.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
LADO = 800
CASILLA = LADO // 8
sys.path.insert(0, str(RAIZ / "scripts"))
from comun import esquinas_por_rejilla  # noqa: E402

CLASES_13 = ["peon_b", "torre_b", "caballo_b", "alfil_b", "dama_b", "rey_b",
             "peon_n", "torre_n", "caballo_n", "alfil_n", "dama_n", "rey_n",
             "vacia"]


def cargar_clasificador(ruta_peso: Path):
    """Carga best_clasificador.pt y devuelve (net, clases, info)."""
    import torch
    from torchvision import models

    ck = torch.load(ruta_peso, map_location="cpu", weights_only=False)
    clases: list[str] = ck["clases"]
    if isinstance(ck["modelo"], dict):
        net = models.mobilenet_v3_small(weights=None)
        net.classifier[3] = __import__("torch").nn.Linear(
            net.classifier[3].in_features, len(clases)
        )
        net.load_state_dict(ck["modelo"])
    else:
        net = ck["modelo"]
    net.eval()
    extras = {clave: ck[clave] for clave in ("epoca", "acc_valid", "tamano")
              if clave in ck}
    return net, clases, extras


def _preparar_lote(imagen):
    """64 tensores, uno por casilla, en el orden del pipeline del cuaderno."""
    import cv2
    import torch
    from torchvision import transforms as T

    transformation = T.Compose([
        T.ToTensor(),
        T.Resize((128, 128), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tiras = []
    for fila in range(8):
        for columna in range(8):
            ya, yb = fila * CASILLA, (fila + 1) * CASILLA
            xa, xb = columna * CASILLA, (columna + 1) * CASILLA
            recorte = imagen[ya:yb, xa:xb]
            tiras.append(transformation(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)))
    return torch.stack(tiras)


def _nombre_casilla(fila: int, columna: int) -> str:
    return f"{chr(ord('a') + columna)}{8 - fila}"


def realinear_por_fichas(imagen, deteccion_funcion) -> tuple[object | None, str]:
    """Vuelve a alinear un recorte con la rejilla que marcan sus fichas.

    Los recortes de T4b heredan el error de esquina del rectificador original:
    son '800x800 pero las fichas cuelgan entre casillas' (se midio ~30-40 px).
    El clasificador necesita piezas CENTRADAS en su casilla, y la rejilla la
    conocemos: la nube de fichas es un 7x7 de pasos, cuyas 4 esquinas son
    exactamente los centros de a8/h8/a1/h1. Warp de esos centros a
    (50,50),(750,50),(750,750),(50,750) y las 64 casillas quedan canonicas.

    Devuelve (imagen_realineada | None, motivo).
    """
    import cv2

    detecciones = deteccion_funcion(imagen)
    conexiones, motivo_detalles = esquinas_por_rejilla(detecciones)
    if conexiones is None:
        return None, f"rejilla de fichas no valida ({motivo_detalles})"

    # conexiones trae las esquinas del TABLERO (0..LADO en el recorte); las
    # mandamos EXACTAMENTE a las esquinas canonicas del cuadrado 800x800.
    destino = np.array([
        [0, 0],
        [LADO - 1, 0],
        [LADO - 1, LADO - 1],
        [0, LADO - 1],
    ], dtype="float32")
    matriz = cv2.getPerspectiveTransform(np.asarray(conexiones, dtype="float32"),
                                         destino)
    alineada = cv2.warpPerspective(
        imagen, matriz, (LADO, LADO), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return alineada, f"realineada: {motivo_detalles}"


def clasificar_recorte(net, clases: list[str], imagen) -> dict[str, dict]:
    """Devuelve {casilla: {'elegida', 'confianza', 'ranking': [(clase, p), ...]}}"""
    import torch

    lote = _preparar_lote(imagen)
    with torch.no_grad():
        salidas = net(lote)
    probs = torch.softmax(salidas, dim=1).numpy()
    resultado: dict[str, dict] = {}
    for i, probs_casilla in enumerate(probs):
        fila, columna = divmod(i, 8)
        orden = np.argsort(probs_casilla)[::-1]
        ranking = [(clases[int(j)], float(probs_casilla[j])) for j in orden]
        resultado[_nombre_casilla(fila, columna)] = {
            "elegida": ranking[0][0],
            "confianza": ranking[0][1],
            "ranking": ranking,
        }
    return resultado


def descartar_fantasmas_baja_confianza(ocupacion: dict[str, dict],
                                       umbral_confianza: float = 0.55,
                                       vacia_minima: float = 0.30) -> list[str]:
    """Quita piezas 'espectro' de suya: fichas inseguras en casillas donde el
    ranking alternativo ya sugiere 'vacia' con fuerza.

    Regla (medida en la revision viva del 2026-11-07): si la casilla proyecta
    una ficha con confianza < umbral y 'vacia' aparece en las 3 primeras
    opciones del ranking con prob >= vacia_minima, la casilla queda vacia.
    Es exactamente el 'N rojo en b8' que fluctua entre frames.

    Modifica los dicts de ocupacion in-place y devuelve las casillas
    cambiadas (para el detalle del conector).
    """
    cambiadas = []
    for casilla, dato in ocupacion.items():
        if dato["elegida"] == "vacia" or dato["confianza"] >= umbral_confianza:
            continue
        ranking = dato.get("ranking") or []
        prob_vacia = next((p for c, p in ranking if c == "vacia"), 0.0)
        if prob_vacia >= vacia_minima:
            dato["_verdict_luz"] = dato["elegida"]
            dato["elegida"] = "vacia"
            dato["confianza_vieja"] = dato["confianza"]
            dato["confianza"] = prob_vacia
            cambiadas.append(casilla)
    return cambiadas


def construir_fen(ocupacion: dict[str, str], turno_blanca: bool = True) -> str:
    """FEN de la posicion con lo que dijo el clasificador (sin filtrar)."""
    import chess

    tablero = chess.Board(None)
    for casilla, clase in ocupacion.items():
        if clase == "vacia":
            continue
        pieza, _, color = clase.partition("_")
        tipo = {
            "peon": chess.PAWN, "torre": chess.ROOK, "caballo": chess.KNIGHT,
            "alfil": chess.BISHOP, "dama": chess.QUEEN, "rey": chess.KING,
        }[pieza]
        tablero.set_piece_at(chess.parse_square(casilla),
                             chess.Piece(tipo, color == "b"))
    tablero.turn = chess.WHITE if turno_blanca else chess.BLACK
    return tablero.fen()


def _violaciones(fen: str) -> str:
    import chess

    return chess.Board(fen).status()


def legalizar(ocupacion: dict[str, dict], turno_blanca: bool = True,
              maximo_intentos: int = 12) -> tuple[str, list[str]]:
    """Post-filtro de movimientos legales sobre las clases del clasificador.

    La posicion inicial/parte legal debe `chess.Board(status)` == LEGAL. Si no,
    se elige la casilla CULPABLE de la primera violacion y se prueba su 2a
    opcion del ranking; se repite hasta MAXIMO intentos.

    Devuelve (fen_final, cambios) donde cambios describe c/GA jury swap.
    """
    import chess

    plano = {c: dto["elegida"] for c, dto in ocupacion.items()}
    cambios: list[str] = []
    for _ in range(maximo_intentos):
        fen = construir_fen(plano, turno_blanca)
        estado = _violaciones(fen)
        # chess.Board.status() devuelve Status.VALID (que NO es None): hay que
        # comparar por valor. Antes comparaba None y el bucle nunca cortaba
        # (agotaba los 12 intentos siempre; funcionaba por pura simetria).
        if estado is None or estado == chess.Status.VALID:
            return fen, cambios

        # Que casilla interviene en la primera violacion?
        culpables = _casillas_culpables(fen, plano)
        for casilla in culpables:
            ranking = ocupacion[casilla]["ranking"]
            actual = plano[casilla]
            dispositivo = [c for c, _ in ranking if c != actual]
            if not dispositivo or dispositivo[0] is None:
                break
            nueva = dispositivo[0]
            plano[casilla] = nueva
            cambios.append(
                f"{casilla}: {actual} -> {nueva} (violacion: {estado.name})"
            )
            break
        else:
            break
    return construir_fen(plano, turno_blanca), cambios


def _casillas_culpables(fen: str, ocupacion: dict[str, str]) -> list[str]:
    """Casillas cuya pieza es sospechosa de la violacion mas evidente.

    Reglas idoma:.quedan: demasiados reyes de un color, peones en las filas
    extremas, 9+ peones en el control del mismo rey del bando, reyes adyacentes.
    Las duas las resuelve el orden del ranking alternativo, no este detector.
    """
    import chess

    tablero = chess.Board(fen)
    sospechosas: list[str] = []
    reyes = [(chess.square_name(c), pieza) for c, pieza in
             [(c, tablero.piece_at(c)) for c in chess.SQUARES]
             if pieza and pieza.piece_type == chess.KING]
    de_mas_por_color = len([1 for _, p in reyes if p.color])
    de_mas_negras = len(reyes) - de_mas_por_color
    if de_mas_por_color > 1 or de_mas_negras > 1:
        sobrante_b = de_mas_por_color - 1
        sobrante_n = de_mas_negras - 1
        for casilla, pieza in reyes:
            if sobrante_b > 0 and pieza.color and sospechosas.count(casilla) < 1:
                sospechosas.append(casilla)
                sobrante_b -= 1
            elif sobrante_n > 0 and not pieza.color:
                sospechosas.append(casilla)
                sobrante_n -= 1

    for casilla, pieza in [(c, tablero.piece_at(c)) for c in chess.SQUARES]:
        if pieza is None:
            continue
        ribete = chess.square_rank(casilla) in (0, 7)
        if pieza.piece_type == chess.PAWN and ribete:
            sospechosas.append(chess.square_name(casilla))

    conteo = Counter(pieza.symbol() for pieza in
                     [tablero.piece_at(c) for c in chess.SQUARES]
                     if pieza and pieza.piece_type == chess.PAWN)
    for simbolo, cantidad in conteo.items():
        if cantidad > 8:
            for c in chess.SQUARES:
                pieza = tablero.piece_at(c)
                if pieza and pieza.piece_type == chess.PAWN and pieza.symbol() == simbolo:
                    sospechosas.append(chess.square_name(c))

    # Reyes adyacentes: el mas debil (menor confianza) lo cambia con su ranking.
    blancas_k, negras_k = [c for c, p in reyes if p.color], \
        [c for c, p in reyes if not p.color]
    if len(blancas_k) == 1 and len(negras_k) == 1:
        fa, ca = divmod(chess.parse_square(blancas_k[0]), 8)
        fn, cn = divmod(chess.parse_square(negras_k[0]), 8)
        if abs(fa - fn) <= 1 and abs(ca - cn) <= 1:
            sospechosas.extend([blancas_k[0], negras_k[0]])

    return list(dict.fromkeys(sospechosas + list(ocupacion)))


def resumen_ocupacion(ocupacion: dict[str, dict]) -> Counter:
    return Counter(dto["elegida"] for dto in ocupacion.values())
