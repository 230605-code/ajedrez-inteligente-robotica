"""Árbitro de juego, reglas de ajedrez y análisis de motor.

Administra el estado de la partida con python-chess, valida jugadas físicas
detectadas por la cámara, diagnostica jugadas ilegales con explicaciones
detalladas en español, descarta cuadros ruidosos por vibración de cámara,
calcula la mejor jugada con Stockfish y genera órdenes para el robot.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import chess
import chess.engine

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from controlador_robot import DespachadorRobot, EventoJugadaRobot  # noqa: E402

PIEZAS_ES = {
    chess.PAWN: "Peón",
    chess.KNIGHT: "Caballo",
    chess.BISHOP: "Alfil",
    chess.ROOK: "Torre",
    chess.QUEEN: "Dama",
    chess.KING: "Rey",
}

CASILLAS = [chess.square_name(c) for c in chess.SQUARES]


def square_to_coords(sq: int) -> tuple[int, int]:
    """Retorna (col, fila) donde a=0..h=7, 1=0..8=7."""
    return chess.square_file(sq), chess.square_rank(sq)


def reconstruir_derechos_enroque(tablero: chess.Board) -> None:
    """Reconstruye derechos de enroque si rey y torres están en casillas iniciales."""
    derechos = []
    for color, primera, letras in ((chess.WHITE, 0, "KQ"), (chess.BLACK, 7, "kq")):
        rey = tablero.piece_at(chess.square(4, primera))
        if not (rey and rey.piece_type == chess.KING and rey.color == color):
            continue
        torre_h = tablero.piece_at(chess.square(7, primera))
        if torre_h and torre_h.piece_type == chess.ROOK and torre_h.color == color:
            derechos.append(letras[0])
        torre_a = tablero.piece_at(chess.square(0, primera))
        if torre_a and torre_a.piece_type == chess.ROOK and torre_a.color == color:
            derechos.append(letras[1])
    tablero.set_castling_fen("".join(derechos) or "-")


@dataclass
class AnalisisMotor:
    mejor_jugada_uci: str = ""
    mejor_jugada_san: str = ""
    origen_sq: int | None = None
    destino_sq: int | None = None
    evaluacion_cp: int = 0
    es_mate: bool = False
    mate_en: int | None = None
    porcentaje_blancas: float = 50.0
    profundidad: int = 0
    pv_uci: list[str] = field(default_factory=list)


@dataclass
class ResultadoJugada:
    tipo: str  # "LEGAL", "ILEGAL", "RUIDO_VISION", "PIEZA_LEVANTADA", "SIN_CAMBIO"
    mensaje: str = ""
    instruccion_correccion: str = ""
    jugada_uci: str = ""
    jugada_san: str = ""
    origen: str = ""
    destino: str = ""
    casillas_afectadas: list[str] = field(default_factory=list)
    evento_robot: EventoJugadaRobot | None = None


class ArbitroAjedrez:
    """Motor de arbitraje, validación de reglas y enlace con Stockfish."""

    def __init__(self, ruta_stockfish: Path | None = None) -> None:
        self.tablero = chess.Board()
        self.ocupacion_base: dict[str, str] = self._obtener_ocupacion_actual()
        self.despachador_robot = DespachadorRobot()
        self.ruta_stockfish = ruta_stockfish or (RAIZ / "tools" / "stockfish" / "stockfish.exe")
        self.motor_stockfish: chess.engine.SimpleEngine | None = None
        self.ultimo_analisis = AnalisisMotor()
        self.estado_partida = "EN_JUEGO"  # "EN_JUEGO", "JAQUE", "JAQUE_MATE", "TABLAS", "BLOQUEADO_ILEGAL"
        self._inicializar_stockfish()
        self.actualizar_analisis_motor()

    def _inicializar_stockfish(self) -> None:
        if self.ruta_stockfish.exists():
            try:
                self.motor_stockfish = chess.engine.SimpleEngine.popen_uci(str(self.ruta_stockfish))
            except Exception as exc:
                print(f"[ARBITRO] No se pudo inicializar Stockfish ({exc})")
                self.motor_stockfish = None

    def cerrar(self) -> None:
        if self.motor_stockfish is not None:
            try:
                self.motor_stockfish.close()
            except Exception:
                pass
            self.motor_stockfish = None

    def __del__(self) -> None:
        self.cerrar()

    def _obtener_ocupacion_actual(self) -> dict[str, str]:
        """Convierte el estado de python-chess al mapa de 13 clases del clasificador."""
        ocupacion = {}
        for sq in chess.SQUARES:
            casilla = chess.square_name(sq)
            p = self.tablero.piece_at(sq)
            if p is None:
                ocupacion[casilla] = "vacia"
            else:
                nombre = {
                    chess.PAWN: "peon", chess.ROOK: "torre", chess.KNIGHT: "caballo",
                    chess.BISHOP: "alfil", chess.QUEEN: "dama", chess.KING: "rey",
                }[p.piece_type]
                color = "b" if p.color == chess.WHITE else "n"
                ocupacion[casilla] = f"{nombre}_{color}"
        return ocupacion

    def iniciar_nueva_partida(self, turno: chess.Color = chess.WHITE) -> str:
        """Reinicia el tablero a la posición estándar inicial con el turno elegido."""
        color_char = "w" if turno == chess.WHITE else "b"
        fen_inicial = f"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR {color_char} KQkq - 0 1"
        self.tablero = chess.Board(fen_inicial)
        self.ocupacion_base = self._obtener_ocupacion_actual()
        self.estado_partida = "EN_JUEGO"
        self.actualizar_analisis_motor()
        return self.tablero.fen()

    def fijar_posicion(self, fen: str, turno: chess.Color | None = None,
                       reconstruir_enroque: bool = False) -> str:
        """Fija la posición activa desde una FEN o lectura de cámara y actualiza Stockfish."""
        board = chess.Board(fen)
        if reconstruir_enroque:
            reconstruir_derechos_enroque(board)
        if turno is not None:
            partes = board.fen().split()
            partes[1] = "w" if turno == chess.WHITE else "b"
            if len(partes) > 3:
                partes[3] = "-"
            board = chess.Board(" ".join(partes))
        self.tablero = board
        self.ocupacion_base = self._obtener_ocupacion_actual()
        self.estado_partida = "EN_JUEGO"
        self._actualizar_estado_fin_partida()
        self.actualizar_analisis_motor()
        return self.tablero.fen()

    def cambiar_turno(self, turno: chess.Color) -> str:
        """Cambia el turno activo y recalcula inmediatamente la mejor respuesta del motor."""
        partes = self.tablero.fen().split()
        partes[1] = "w" if turno == chess.WHITE else "b"
        if len(partes) > 3:
            partes[3] = "-"
        self.tablero = chess.Board(" ".join(partes))
        self.ocupacion_base = self._obtener_ocupacion_actual()
        self._actualizar_estado_fin_partida()
        self.actualizar_analisis_motor()
        return self.tablero.fen()

    def _actualizar_estado_fin_partida(self) -> None:
        if self.tablero.is_checkmate():
            self.estado_partida = "JAQUE_MATE"
        elif self.tablero.is_stalemate() or self.tablero.is_insufficient_material() or self.tablero.can_claim_draw():
            self.estado_partida = "TABLAS"
        elif self.tablero.is_check():
            self.estado_partida = "JAQUE"
        else:
            self.estado_partida = "EN_JUEGO"

    def actualizar_analisis_motor(self, tiempo_limite: float = 0.25) -> AnalisisMotor:
        """Pide a Stockfish evaluar la posición y proponer la mejor respuesta."""
        analisis = AnalisisMotor()
        if self.motor_stockfish is None or self.tablero.is_game_over():
            self.ultimo_analisis = analisis
            return analisis

        try:
            tablero_eval = chess.Board(self.tablero.fen())
            resultado = self.motor_stockfish.analyse(tablero_eval, chess.engine.Limit(time=tiempo_limite))
            pv = resultado.get("pv", [])
            if pv:
                mejor = pv[0]
                analisis.mejor_jugada_uci = mejor.uci()
                analisis.mejor_jugada_san = tablero_eval.san(mejor)
                analisis.origen_sq = mejor.from_square
                analisis.destino_sq = mejor.to_square
                analisis.pv_uci = [m.uci() for m in pv]

            score = resultado.get("score")
            if score is not None:
                score_blancas = score.white()
                if score_blancas.is_mate():
                    analisis.es_mate = True
                    analisis.mate_en = score_blancas.mate()
                    analisis.evaluacion_cp = 10000 if (analisis.mate_en or 1) > 0 else -10000
                    analisis.porcentaje_blancas = 100.0 if (analisis.mate_en or 1) > 0 else 0.0
                else:
                    cp = score_blancas.score() or 0
                    analisis.evaluacion_cp = cp
                    analisis.es_mate = False
                    analisis.porcentaje_blancas = round(100.0 / (1.0 + math.pow(10, -cp / 400.0)), 1)

            analisis.profundidad = resultado.get("depth", 0)
        except (chess.engine.EngineTerminatedError, chess.engine.EngineError) as exc:
            print(f"[ARBITRO] Stockfish terminó o falló ({exc}). Reinicializando motor...")
            self.cerrar()
            self._inicializar_stockfish()
        except Exception as exc:
            print(f"[ARBITRO] Error al analizar con Stockfish: {exc}")

        self.ultimo_analisis = analisis
        return analisis

    def deducir_y_validar_movimiento(self, lectura_actual: dict[str, str]) -> ResultadoJugada:
        """Compara la ocupación física detectada contra la base confirmada."""
        if self.tablero.is_game_over():
            return ResultadoJugada(tipo="FIN_PARTIDA",
                                   mensaje=f"La partida ha concluido ({self.estado_partida}). Reinicia o carga nueva posición.")

        base = self.ocupacion_base

        origenes = sorted(c for c in CASILLAS
                          if base.get(c, "vacia") != "vacia" and lectura_actual.get(c, "vacia") == "vacia")
        destinos = sorted(c for c in CASILLAS
                          if lectura_actual.get(c, "vacia") != "vacia" and lectura_actual.get(c) != base.get(c))

        # Sin diferencias con respecto a la base confirmada
        if not origenes and not destinos:
            if self.estado_partida == "BLOQUEADO_ILEGAL":
                self.estado_partida = "EN_JUEGO"
                self._actualizar_estado_fin_partida()
            return ResultadoJugada(tipo="SIN_CAMBIO", mensaje="Tablero coincide exactamente con la posición activa.")

        # Caso: Pieza(s) levantada(s) en mano o captura en proceso (1 o 2 vaciadas, sin destino aún)
        if len(origenes) in (1, 2) and len(destinos) == 0:
            cas_str = ", ".join(origenes)
            return ResultadoJugada(tipo="PIEZA_LEVANTADA", origen=cas_str,
                                   mensaje=f"Pieza(s) levantada(s) en {cas_str}. Esperando colocación.")

        # Caso: Ruido / destello aislado (0 orígenes, 1 destino)
        if len(origenes) == 0 and len(destinos) == 1:
            return ResultadoJugada(tipo="RUIDO_VISION",
                                   mensaje=f"Variación aislada en {destinos[0]} descartada por ruido óptico.")

        # Caso 1: Movimiento simple o captura (1 casilla vaciada, 1 casilla ocupada)
        if len(origenes) == 1 and len(destinos) == 1:
            orig = origenes[0]
            dest = destinos[0]
            return self._juzgar_movimiento_simple(orig, dest, base, lectura_actual)

        # Caso 2: Enroque (2 casillas vaciadas, 2 casillas ocupadas: rey + torre)
        if len(origenes) == 2 and len(destinos) == 2:
            return self._juzgar_enroque(origenes, destinos, base, lectura_actual)

        # Caso 3: Captura al paso (2 casillas vaciadas: peón atacante y peón capturado; 1 destino)
        if len(origenes) == 2 and len(destinos) == 1:
            return self._juzgar_al_paso(origenes, destinos[0], base, lectura_actual)

        # Caso 4: Ruido masivo de visión (cámara movida, sombra de mano, luz ambiental)
        # NUNCA se trata como jugada ilegal; se descarta silenciosamente para proteger la partida
        total_cambios = len(origenes) + len(destinos)
        return ResultadoJugada(
            tipo="RUIDO_VISION",
            mensaje=(f"Cuadro descartado: {total_cambios} casillas alteradas simultáneamente "
                     f"(posible movimiento de cámara o sombra). Manteniendo posición confirmada."),
            casillas_afectadas=origenes + destinos,
        )

    def _juzgar_movimiento_simple(self, orig: str, dest: str, base: dict[str, str],
                                  actual: dict[str, str]) -> ResultadoJugada:
        sq_orig = chess.parse_square(orig)
        sq_dest = chess.parse_square(dest)
        pieza = self.tablero.piece_at(sq_orig)

        # Promoción automática si un peón llega al final
        promo = None
        if pieza and pieza.piece_type == chess.PAWN and chess.square_rank(sq_dest) in (0, 7):
            clase_dest = actual.get(dest, "")
            promo_tipo = clase_dest.split("_")[0] if "_" in clase_dest else "dama"
            promo = {
                "dama": chess.QUEEN, "torre": chess.ROOK,
                "alfil": chess.BISHOP, "caballo": chess.KNIGHT
            }.get(promo_tipo, chess.QUEEN)

        move = chess.Move(sq_orig, sq_dest, promotion=promo)

        if move in self.tablero.legal_moves:
            return self._aplicar_jugada_legal(move, orig, dest, pieza)
        else:
            self.estado_partida = "BLOQUEADO_ILEGAL"
            explicacion = self._explicar_ilegalidad(move, orig, dest, pieza)
            nombre_p = PIEZAS_ES.get(pieza.piece_type, "Pieza") if pieza else "Pieza"
            instruccion = f"Regresa físicamente el {nombre_p} de '{dest}' a '{orig}' para reanudar el juego."
            return ResultadoJugada(
                tipo="ILEGAL",
                mensaje=explicacion,
                instruccion_correccion=instruccion,
                jugada_uci=move.uci(),
                origen=orig,
                destino=dest,
                casillas_afectadas=[orig, dest],
            )

    def _juzgar_enroque(self, origenes: list[str], destinos: list[str],
                        base: dict[str, str], actual: dict[str, str]) -> ResultadoJugada:
        reyes = [o for o in origenes if base[o].startswith("rey_")]
        if len(reyes) == 1:
            rey_orig = reyes[0]
            rey_dest_candidatos = [d for d in destinos if chess.square_file(chess.parse_square(d)) in (2, 6)]
            if rey_dest_candidatos:
                rey_dest = rey_dest_candidatos[0]
                sq_orig = chess.parse_square(rey_orig)
                sq_dest = chess.parse_square(rey_dest)
                move = chess.Move(sq_orig, sq_dest)
                if move in self.tablero.legal_moves:
                    p = self.tablero.piece_at(sq_orig)
                    return self._aplicar_jugada_legal(move, rey_orig, rey_dest, p, es_enroque=True)
                else:
                    self.estado_partida = "BLOQUEADO_ILEGAL"
                    inst = "El rey no puede enrocar en esta posición. Regresa rey y torre a sus casillas iniciales."
                    return ResultadoJugada(
                        tipo="ILEGAL",
                        mensaje="Intento de enroque ilegal en la posición actual.",
                        instruccion_correccion=inst,
                        casillas_afectadas=origenes + destinos,
                    )

        # No hay rey involucrado: 2 piezas cambiaron a la vez por vibración de cámara o ruido
        return ResultadoJugada(
            tipo="RUIDO_VISION",
            mensaje="Dos piezas cambiadas simultáneamente sin rey. Descartado como ruido de visión.",
            casillas_afectadas=origenes + destinos,
        )

    def _juzgar_al_paso(self, origenes: list[str], dest: str, base: dict[str, str],
                        actual: dict[str, str]) -> ResultadoJugada:
        clase_llega = actual.get(dest, "")
        if not clase_llega.startswith("peon_"):
            return ResultadoJugada(tipo="RUIDO_VISION", mensaje="Cuadro descartado: patrón no coincide con captura al paso.")

        mueven = [o for o in origenes if base[o] == clase_llega]
        if len(mueven) == 1:
            peon_orig = mueven[0]
            sq_orig = chess.parse_square(peon_orig)
            sq_dest = chess.parse_square(dest)
            move = chess.Move(sq_orig, sq_dest)
            if move in self.tablero.legal_moves:
                p = self.tablero.piece_at(sq_orig)
                return self._aplicar_jugada_legal(move, peon_orig, dest, p, es_al_paso=True)

        return ResultadoJugada(tipo="RUIDO_VISION", mensaje="Patrón de dos orígenes no válido para captura al paso.")

    def _aplicar_jugada_legal(self, move: chess.Move, orig: str, dest: str, pieza: chess.Piece | None,
                              es_enroque: bool = False, es_al_paso: bool = False) -> ResultadoJugada:
        san = self.tablero.san(move)
        uci = move.uci()
        es_captura = self.tablero.is_capture(move) or es_al_paso
        color_str = "blanco" if self.tablero.turn == chess.WHITE else "negro"
        nombre_pieza = PIEZAS_ES.get(pieza.piece_type if pieza else chess.PAWN, "Peón").lower()

        torre_orig, torre_dest = None, None
        if es_enroque:
            if dest in ("g1", "g8"):
                torre_orig = "h1" if dest == "g1" else "h8"
                torre_dest = "f1" if dest == "g1" else "f8"
            else:
                torre_orig = "a1" if dest == "c1" else "a8"
                torre_dest = "d1" if dest == "c1" else "d8"

        self.tablero.push(move)
        self.ocupacion_base = self._obtener_ocupacion_actual()
        self._actualizar_estado_fin_partida()
        self.actualizar_analisis_motor()

        evento_robot = EventoJugadaRobot(
            uci=uci,
            san=san,
            origen=orig,
            destino=dest,
            origen_col_fila=square_to_coords(move.from_square),
            destino_col_fila=square_to_coords(move.to_square),
            pieza=nombre_pieza,
            color=color_str,
            es_captura=es_captura,
            casilla_captura=dest if not es_al_paso else (dest[0] + orig[1]),
            es_enroque=es_enroque,
            torre_origen=torre_orig,
            torre_destino=torre_dest,
            es_promocion=move.promotion is not None,
            pieza_promocion=PIEZAS_ES.get(move.promotion, "dama").lower() if move.promotion else None,
            fen_resultante=self.tablero.fen(),
            timestamp=time.time(),
        )

        self.despachador_robot.despachar(evento_robot)

        return ResultadoJugada(
            tipo="LEGAL",
            mensaje=f"Jugada legal aceptada: {san} ({uci})",
            jugada_uci=uci,
            jugada_san=san,
            origen=orig,
            destino=dest,
            casillas_afectadas=[orig, dest],
            evento_robot=evento_robot,
        )

    def _explicar_ilegalidad(self, move: chess.Move, orig: str, dest: str,
                             pieza: chess.Piece | None) -> str:
        """Genera una explicación pedagógica en español."""
        if pieza is None:
            return f"No hay ninguna pieza en la casilla de origen '{orig}'."

        if pieza.color != self.tablero.turn:
            bando_pieza = "blanca" if pieza.color == chess.WHITE else "negra"
            turno_actual = "blancas" if self.tablero.turn == chess.WHITE else "negras"
            return f"Moviste una pieza {bando_pieza}, pero el turno actual corresponde a las {turno_actual}."

        dest_sq = move.to_square
        pieza_dest = self.tablero.piece_at(dest_sq)
        if pieza_dest and pieza_dest.color == pieza.color:
            nombre_propia = PIEZAS_ES.get(pieza_dest.piece_type, "pieza").lower()
            return f"La casilla de destino '{dest}' ya está ocupada por una pieza propia ({nombre_propia})."

        if self.tablero.is_pinned(pieza.color, move.from_square):
            return f"El {PIEZAS_ES.get(pieza.piece_type, 'pieza')} en '{orig}' está clavado defendiendo a su rey de un jaque."

        if self.tablero.is_check():
            return f"Tu rey está en JAQUE; el movimiento de '{orig}' a '{dest}' no elimina la amenaza."

        if pieza.piece_type == chess.KING and self.tablero.is_attacked_by(not pieza.color, dest_sq):
            return f"El rey no puede moverse a '{dest}' porque esa casilla se encuentra amenazada (quedaría en jaque)."

        if pieza.piece_type == chess.PAWN:
            col_orig, fila_orig = square_to_coords(move.from_square)
            col_dest, fila_dest = square_to_coords(dest_sq)
            delta_fila = fila_dest - fila_orig if pieza.color == chess.WHITE else fila_orig - fila_dest
            if delta_fila < 0:
                return "Los peones no pueden retroceder."
            if col_orig != col_dest and pieza_dest is None:
                return f"Un peón solo avanza en diagonal si captura una pieza enemiga en '{dest}'."
            if col_orig == col_dest and pieza_dest is not None:
                return f"El peón está bloqueado y no puede capturar hacia el frente en '{dest}'."

        nombre_p = PIEZAS_ES.get(pieza.piece_type, "La pieza")
        return f"{nombre_p} no puede desplazarse de '{orig}' a '{dest}' según las reglas de movimiento."
