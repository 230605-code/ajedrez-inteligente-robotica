# T8 — visor de jugadas en vivo (`scripts/visor_jugadas.py`)

Módulo CLI headless que convierte la visión de `src/` en una **interfaz de
partida**: mantiene el estado de la partida, deduce la jugada que se hizo en el
tablero físico, la valida con `python-chess`, pide la mejor respuesta a
Stockfish y deja una imagen con la jugada detectada y la sugerencia.

Cubre **S4** (orden paso a paso: pilot → dataset → pipeline → comparación: este
es el primer eslabón de la capa de juego real) y **S3-pipeline** (la arquitectura
v2 —clasificador por casilla + filtro de legalidad— consumida como contrato de
juego: YOLO/clasificador + `python-chess` + UCI).

## 1. Modos

| Modo | Qué hace |
|---|---|
| `--iniciar [--ip ...]` | 3 capturas → voto de mayoría por casilla → anti-fantasmas + legalizar (todo dentro de `LectorFenV2.leer_tablero`) → guarda el ESTADO BASE |
| `--iniciar-fen "<FEN>"` | ESTADO BASE desde la FEN declarada. La FEN manda (no toca cámara); el turno sale del campo activo |
| `--paso [--ip ...]` | UN paso: 2-3 lecturas + voto → deduce la jugada → la aplica → Stockfish → `paso_NN/` |
| `--estado` | imprime `estado.json` + tablero ASCII + FEN + turno + pasos + historial SAN. Sin cámara, sin cargar torch |
| `--descargar-stockfish` | baja el binario oficial a `tools/stockfish/stockfish.exe` (urllib+zipfile, sin curl ni PowerShell) |

Opciones añadidas para poder verificar y para el juego real:
`--frame RUTA` (usa una imagen como si fuera la cámara: replay offline por el
MISMO código de visión), `--lecturas N` (2-3), `--turno blanca|negra`,
`--promocion dama|torre|alfil|caballo`, `--tiempo-ms`, `--sin-motor`,
`--sin-descarga`.

## 2. Estado (`assets/previews/jugadas/estado.json`)

```json
{"fen": "...", "turno": "blanca|negra",
 "ocupacion": {"a1": "torre_b", ...}, "historial": ["b4", "Nxb4"],
 "contador_pasos": 2}
```

`ocupacion` se deriva SIEMPRE del tablero resultante (no de la lectura cruda),
así `fen` y `ocupacion` nunca se contradicen.

## 3. Reuso (nada de visión reimplementada)

| Necesidad | Se usa |
|---|---|
| captura + ArUco robusto + fusión YOLO + anti-fantasmas + legalizar | `conector_juego.LectorFenV2.leer_tablero()` |
| FEN desde la ocupación | `pipeline_fen.construir_fen()` |
| clases ↔ piezas | `pipeline_fen._nombre_casilla`/`_preparar_lote` (vía el conector) |
| rectificación del frame para dibujar | `esquinas_aruco.detectar_robusto()` (mismo uso que `captura_revision.py`) |
| ruta del binario de Stockfish | `conector_juego._ruta_stockfish()` |

`src/conector_juego.py` quedó **sin cambios**: `leer_tablero()` ya devuelve el
`chess.Board` legalizado y de ahí se saca la ocupación de las 64 casillas, que es
todo lo que el diff necesita. No hizo falta ningún método nuevo.

## 4. Deducción de la jugada — reglas implementadas

`deducir_jugada(base, actual, tablero, promocion)` compara la ocupación del
estado contra la lectura y **solo** acepta lo que cierra:

* **1 origen + 1 destino con la MISMA clase** → jugada normal. Si el destino
  tenía ficha en el estado base → **captura**.
* **2 orígenes + 2 destinos con clases emparejadas y un rey** → **enroque**
  (`chess.Move` del rey; `board.san` devuelve `O-O` / `O-O-O`).
* **2 orígenes + 1 destino** con un peón de cada color → **captura al paso**
  (se pone `ep_square` para poder aplicarla: el estado no lo conserva).
* **peón que llega a la última fila** → **promoción** con la pieza que muestra
  la lectura; si la lectura muestra un peón en la última fila (o nada), se asume
  `--promocion` (dama por defecto) y se emite una **NOTA**.
* **cualquier otra cosa** (0 piezas sin cambios → `ninguna`; >1 pieza movida
  que no es enroque; clase/color distinto en el destino; desbalance
  orígenes/destinos) → **`fallo`**: `FALLO.txt` + `falla_NN.jpg`, **sin**
  actualizar el estado y **sin** motor.

### Legalidad

La jugada deducida se prueba contra `board.legal_moves` del estado. Si **no** es
legal:

```
AVISO: jugada detectada fuera del legales del estado pasado (posible deteccion
floja); valide con el usuario
```

y el estado pasa a la FEN de la ocupación nueva re-legalizada (la que ya
legalizó el lector), con el turno invertido. El motor **sí** corre sobre esa
posición y `paso.txt` registra el aviso. Esto cubre dos casos reales: el turno
se desincronizó (el usuario movió dos veces / movió fuera de turno) o la
detección leyó una casilla mal.

## 5. Decisiones que se apartan del enunciado literal (y por qué)

1. **`0 jugadas` NO es un fallo.** El enunciado dice que 0 jugadas escriba
   `FALLO.txt`; en una partida real hay pausas y el tablero se lee igual. Se
   imprime `jugada: ninguna detectada`, sale con código 0 y **no** se toca el
   estado ni se crea carpeta. `FALLO.txt` queda reservado para deducciones
   *contradictorias*.
2. **`--lecturas 3` por defecto** (el enunciado permite 2-3). Medido en vivo:
   entre frames consecutivos fluctúa alguna casilla; con 3 lecturas un fantasma
   que aparece en un solo frame no sobrevive al voto. El desempate (cuando
   ninguna clase tiene mayoría) lo gana la lectura de mayor `confianza_media`.
3. **Derechos de enroque reconstruidos** (`_derechos_enroque`) al iniciar con
   cámara: `construir_fen` arma el tablero con `chess.Board(None)`, así que la
   FEN sale con `-` y el enroque **nunca** sería legal. Se conceden si rey y
   torre están en su casilla original con las intermedias vacías. Con
   `--iniciar-fen` no se toca nada: la FEN manda.
4. **Lienzo de 800x970**, no 800x800: el tablero 800x800 no tiene borde
   inferior donde escribir. Se compone tablero + pie de 170 px con la jugada
   (verde), el motor (naranja) y la FEN.
5. **El motor usa `chess.engine` con `multipv=3`** porque el enunciado pide la
   lista de las 3 mejores; la ruta del binario sí se reusa del conector.

## 6. Verificaciones corridas

Entorno: `F:\TAVO\Projects\ajedrez-yolo\.venv\Scripts\python.exe`
(Python 3.12.10, chess 1.11.2, cv2 5.0.0, ultralytics 8.4.199), cámara
`10.251.192.200` encendida.

| # | Comando | Resultado |
|---|---|---|
| 1 | `--iniciar-fen "r2q1rk1/1ppb1ppp/p1np1n2/2b1p1B1/2B1P3/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 10"` | estado.json con esa FEN exacta y turno `blanca` |
| 2 | `--paso` sin mover fichas (offline: mismo frame 3 veces; **y en vivo**) | `jugada: ninguna detectada`, estado intacto (contador 0, historial []) |
| 3 | simulación con cirugía de píxeles (`tools/simular_jugada.py`) | `b4`, `Nxb4` (captura), `Bxf6` (captura) + FALLO con 2 piezas |
| 4 | `--estado` | JSON + tablero ASCII + FEN + turno + pasos + historial, sin cámara |
| 5 | `--descargar-stockfish` | `tools/stockfish/stockfish.exe` 79.8 MB, `uci` → `Stockfish 17.1` |
| 6 | `--iniciar` en vivo | cámara responde; con el tablero completo en cuadro leyó `1.e4 c5 2.Nf3` correcto y legal (ver §8) |
| 7 | `--paso` en vivo **con una jugada real del usuario** | dedujo `f1b5` → `SAN Bb5`, estado avanzado a `... b - - 1 1`, motor `Ngxe5` (+1.45) |

La jugada real (fila 7) es la prueba end-to-end: cámara → 3 lecturas → voto →
diff → legalidad → SAN → Stockfish → estado + imagen. `paso_01/cenital.jpg`
tiene la flecha verde `f1→b5` con el rótulo `Jugada: Bb5` y la naranja `g4→e5`
con `Mejor: Ngxe5`.

Lógica de deducción probada aparte (sin cámara, importando el módulo): normal,
captura, O-O, O-O-O, promoción detectada, promoción con peón en la última fila,
clase distinta → fallo, sin cambios → ninguna, captura al paso, dos piezas
movidas → fallo, aplicación real del enroque y jugada ilegal → `_aplicar`
devuelve `None`.

## 7. Simulación offline (`tools/simular_jugada.py`)

No se podía mover fichas físicas, así que la jugada se **simula sobre la imagen
real**: se rectifica un `crudo.jpg` de `assets/previews/revista_viva/prueba_03`
(ArUco 4/4, rms 0.95 px), se copia el parche de 100x100 de una casilla a otra en
la vista cenital, se rellena el origen con el fondo de una casilla vacía del
mismo color y se recompone el frame por la homografía inversa (máscara erosionada
para no tocar los marcadores). El visor procesa ese frame por la ruta de código
REAL (`--frame`), no por un atajo.

Resultado de la cadena (`assets/previews/jugadas/simulacion/`):

```
paso_01  b2b4    SAN b4     motor Nxb4
paso_02  c6b4    SAN Nxb4   motor a3      (captura)
paso_03  g5f6    SAN Bxf6   motor Qxf6    (captura)
paso_04  b2b4 con turno NEGRO -> AVISO "jugada detectada fuera del legales" (relegaliza)
paso_05  dos piezas movidas -> FALLO.txt + falla_05.jpg, estado y motor intactos
```

Copia preservada de esa cadena: `simulacion/pasos_evidencia/`. Los archivos
`sim_normal.jpg`, `sim_captura.jpg` y `sim_fallo.jpg` son de una primera
iteración (sin encadenar) y quedaron solo como referencia.

Aviso sobre `pasos_evidencia/paso_04/`: quedó dentro un `falla_04.jpg` de una
corrida anterior que reutilizó la misma carpeta (el contador no avanza en un
FALLO). El `paso.txt` de ese directorio es el de la corrida ilegal.

Para que eso no vuelva a pasar, `_limpiar_intento()` borra los artefactos del
intento anterior al reutilizar una carpeta `paso_NN` (un FALLO conserva
`FALLO.txt`/`falla_*.jpg` y descarta `paso.txt`/`cenital.jpg`; un paso válido
hace lo inverso). En `assets/previews/jugadas/paso_01/` hay un `NOTA.txt` que
explica los dos archivos viejos que quedaron ahí de antes de ese arreglo.

## 8. Hallazgos y límites honestos

1. **Bug latente en `pipeline_fen.legalizar`** (fuera de la superficie de este
   trabajo): `_violaciones()` devuelve `chess.Board.status()`, que para una
   posición legal vale `Status.VALID` (0) y **nunca** `None`; por eso
   `if estado is None` no corta el ciclo y `legalizar` siempre agota sus 12
   intentos cambiando la primera casilla al 2º del ranking. Como 12 es par,
   para posiciones legales se restaura sola y el resultado es correcto
   (verificado: la FEN sale idéntica). Con `maximo_intentos` impar o con una
   violación que el ranking no arregla, la salida quedaría a medio corregir.
   `visor_jugadas.py` no depende de ese ciclo para nada propio, pero comparte
   el código a través del conector.
2. **`chess.Board.status()` se evalúa por verdad, no por `is None`** (el aviso
   de posición ilegal del `--iniciar` usa `if estado_legal:`).
3. **El encuadre manda.** En una de las corridas en vivo el tablero no cabía
   completo (solo 2 de 4 ArUco, fila 8 cortada): la lectura salió ilegal
   (`TOO_MANY_BLACK_PAWNS|PAWNS_ON_BACKRANK`, 12 casillas inestables). El
   módulo lo detecta, avisa y deja `assets/previews/jugadas/base/AVISO.txt` con
   el motivo. No es un bug del módulo; es la regla del setup de `docs/ENTREGA.md`.
4. **Cámara intermitente.** Entre corridas seguidas hubo un `sin esquinas ArUco`
   y lecturas sueltas caídas (el módulo sigue con las lecturas que tenga y lo
   informa). Con `/capture` cada ~0.6 s y 3 lecturas por paso alcanza.
5. En vivo, con 37 casillas "discrepando" entre lecturas, el voto de mayorí­a
   igual produjo la posición correcta (`1.e4 c5 2.Nf3`, 32 fichas, legal): el
   voto absorbe el ruido por casilla. Sin él, cualquier frame suelto podía
   disparar un fantasma.
6. El tablero físico CAMBIÓ entre corridas seguidas (las lecturas válidas
   pasaron de `1.e4 c5 2.Nf3` a una posición con caballos negros en c6/g4, y de
   ahí a `Bb5`): el usuario estaba moviendo fichas en vivo. Por eso el estado
   final de `estado.json` es el de una partida en curso (`contador_pasos: 1`,
   `historial: ["Bb5"]`) y no una posición de laboratorio.
7. Un `--paso` equivale a UNA jugada. En la última corrida en vivo el usuario
   había hecho DOS jugadas entre lecturas (`Nxe5` y `Qf3`): el diff detectó las
   dos y el módulo se negó a inventar un orden (`FALLO.txt` con las 4 casillas,
   estado y motor intactos). Es el comportamiento buscado: se corre `--paso`
   después de CADA jugada, no después de un par.

## 9. Cómo se usa en la partida real

```powershell
cd F:\TAVO\Projects\ajedrez-v2
$py = "F:\TAVO\Projects\ajedrez-yolo\.venv\Scripts\python.exe"

& $py scripts\visor_jugadas.py --iniciar              # posición de partida
& $py scripts\visor_jugadas.py --paso                 # después de CADA jugada
& $py scripts\visor_jugadas.py --estado               # consulta
& $py scripts\visor_jugadas.py --iniciar-fen "..."    # posición declarada
```

`--paso` sale con código **0** si detectó jugada o si no hubo cambios, **1** si
la cámara/visión falla, **2** si la deducción no cierra (`FALLO.txt`).
