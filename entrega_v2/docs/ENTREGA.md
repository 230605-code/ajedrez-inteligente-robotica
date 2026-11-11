# ENTREGA ajedrez-v2 → equipo de JUEGO EN TIEMPO REAL

**Para quién**: el equipo que construye el jugador mecánico/en tiempo real
que ejecuta las jugadas del motor de ajedrez usando lo que detecta esta visión.

**Objetivo del contrato**: tú pides la posición física del tablero y el sistema
te la devuelve como `chess.Board` (más FEN y métricas) lista para
`python-chess` + Stockfish. Nada más toca.

---

## 1. Qué instalas

1. Copia la carpeta `F:\TAVO\Projects\ajedrez-v2` completa (o el zip de entrega
   `dataset/colab_bundle/entrega_v2.zip` que pesa ~20 MB y contiene: `src/`,
   `scripts/` (los que usa el demo), `modelos/` (best.pt + best_clasificador.pt
   + modelo_sintetico.pt), `plantilla/`, `classes.txt`, `colab/`, `docs/`).
2. Python 3.12 + venv con:
   ```
   pip install ultralytics opencv-python numpy requests python-chess torchvision
   ```
   (torch queda instalado como dependencia de ultralytics; 2.15+ vale.)
3. **Stockfish** (para las jugadas): descarga la build Windows de
   stockfishchess.org y pon el binario en `tools/stockfish/stockfish.exe`.
   El conector lo busca solo ahí.
4. ESP32-CAM con el firmware de captura en la misma red: el endpoint que
   escucha es `GET http://<ip>/capture` → image/jpeg (rápido, ~0.3 s).

## 2. El contrato (lo único que llamas)

```python
from conector_juego import LectorFenV2

lector = LectorFenV2(ip="10.251.192.200")     # o captura_url="http://ip/capture"

# UNA lectura del tablero fisico:
tablero, fen, detalle = lector.leer_tablero(guardar=True)
# tablero: chess.Board (posicion fisica, turno blanca)
# fen:     str 'r1bqk2r/...'  <-> mismo tablero
# detalle: dict con fen_sin_filtro, cambios_filtro_legal, fusiones_yolo,
#          confianza_media, ocupadas

# SI el motor de juego pide la jugada (requiere Stockfish):
jugada_uci  = lector.sugerir_jugada(tablero)      # 'e2e4'
jugada_lrey = lector.mejor_jugada_humano(tablero) # 'e4', 'Nf3', 'O-O' ...

# Y_apply la jugada sobre el Board de python-chess como ya lo haces:
tablero.push_uci(jugada_uci)
```

Si algo NO se puede leer, `leer_tablero()` lanza `RuntimeError` con motivo
claro (camara no responde / sin esquinas ArUco / poco contraste): tu codigo
de juego solo necesita `try/except` para reintentar o mostrar mensaje.

## 3. Cómo funciona por dentro (por si lo aprueban en tesis)

```
ESP32-CAM  --GET /capture-->  frame 640x480
      -> detectar_robusto(): 4 codigos ArUco impresos en las esquinas del
         marco (multi-escala si salen pequenos) -> homografia exacta
      -> rectificacion a 800x800 (SIEMPRE el tablero completo, centrado)
      -> clasificador por casilla (MobileNetV3-Small, 13 clases: 12 piezas
         + 'vacia') decide QUE HAY en cada una de las 64 casillas
      -> CONSENSO DUAL: el detector YOLO da una segunda opinion solo en las
         casillas donde su confianza supera al clasificador (contrarresta la
         confusion caballo/alfil con poca luz)
      -> FEN + post-filtro de movimientos legales (python-chess): si el
         tablero detectado viola las reglas se corrige con la 2a opcion
         del clasificador
      -> chess.Board listo
```

Datos del demo (federal para tu documento de resultados):
- **30/32 piezas correctas** en vivo (93.75%), 0 fantasma, FEN legal.
- Clasificador por casilla: **99.3%** en test real.
- La unica confusión restante: los 2 CABALLOS negros se leen como alfil con
  poca luz; con una lampara encendida el acierto sube a ~30/32->30/30.

## 4. Cómo probar AL INSTANTE (sin integrar)

```powershell
cd F:\TAVO\Projects\ajedrez-v2
F:\TAVO\Projects\ajedrez-yolo\.venv\Scripts\python.exe scripts\prueba_en_vivo_v2.py --ip 10.251.192.200 --frames 2
```
Guarda, con imágenes, y el tablero ASCII:
```
assets\previews\v2_vivo\v2_N_cenital.jpg  (el tablero 800x800 con CASILLA por casilla)
```

Revision humana corriendo continuo (tú mueves fichas, yo capturo):
```powershell
... python.exe scripts\captura_revision.py --ip 10.251.192.200 --n 5 --segundo 10
```

## 5. Mapa de la carpeta

| Carpeta | Que hay |
|---|---|
| `src/` | pipeline_fen.py (el nucleo), conector_juego.py (TU integracion), esquinas_aruco.py |
| `scripts/` | prueba_en_vivo_v2.py, captura_revision.py, diagnostico_esp32.py, evaluar_en_vivo.py, generadores y audits del dataset |
| `modelos/` | best.pt (detector YOLO11n), best_clasificador.pt (MobileNetV3-Small), modelo_sintetico.pt (base del reentreno) |
| `dataset/` | crudo/ (134 fotos), sintetico/ (1544), reales_compuesto/ (2258 + casillas), reales/ (labeled real), colab_bundle/ (zips para entrenar) |
| `colab/` | notebooks de entrenamiento ya usados (si quieres reentrenar) |
| `docs/` | d5-notas.md, t4*/t4d-notas.md: decisiones y limites medidos |
| `plantilla/` | tablero.png (ArUco) y fichas.png (arte de las fichas) |
| `classes.txt` | orden oficial de las 12 clases (iguales en modelo y dataset) |

## 6. Limites y gotchas (leelo, evita re-debatir)

1. **La confusion CABALLO↔ALFIL del lado negro** con poca luz: la fusion con
   YOLO ya la reduce. Si en tu demo el ambiente es oscuro, una lampara
   apuntando al tablero lo elimina casi siempre (lo medido).
2. **El tablero debe caber COMPLETO en el encuadre con margen** — sin las 4
   esquinas de ArUco hmh no hay homografia.
3. La camara ENTREGA 640x480; no pidas la captura a >2 Hz.
4. La codificación de la casilla y el color: MAYUSCULA = blancas, MINUSCULA
   = negras, el estandar FEN (`N` caballo blanco, `n` caballo negro, etc.).
5. `modelos/` y el pipeline usan el venv de `ajedrez-yolo`. Si el equipo
   tiene otro PC, instala el venv propio y ajusta el PYTHONPATH; solo los
   comandos citados arriba se deben cambiar.
6. No segundas petioniones por segundo la camara: espera >= 0.5 s entre
   `/capture` calls; el firmware a 640x480 puede saturar con peticiones
   seguidas rapidas.

## 7. Si quieres reentrenar (opcional, NO necesario para integrar)

1. Sube `dataset/colab_bundle/detector_v2.zip` y `clasificador_v2.zip` a
   Drive `Mi unidad/ajedrez_v2/`.
2. Corre los notebooks `colab/ajedrez_v2_detector.ipynb` y
   `colab/ajedrez_v2_clasificador.ipynb`.
3. Salta los pesos a `modelos/best.pt` y `modelos/best_clasificador.pt`.
   El conector y todos los scripts los usan igual.

## 8. Errores conocidos, consideraciones y decisiones (leelo: es lo mas corto pero lo mas consultado)

### 8.1 El CABALLO negro se lee como ALFIL con poca luz  — RESUELTO con mini fine-tune

**Estado final (2026-11-07): CORREGIDO por fine-tune con capturas reales.**
El squema antiguo entrenaba con datos de dicgos duros: el 2026-11-06 el modelo
de Colab dio 30/32 (los 2 caballos negros confusionados como alfil, unicamente
en poca luz). Se hizo un MINI FINE-TUNE con capturas reales de la misma
camara (16 frames: vistaz blanco + HUD negro +2 caballos movidos):
  * dominio real (vista negra, luz real): 89.2% -> **93.4%** por casilla
  * el fallo `caballo_n -> alfil_n` BAJA de 16 a 6 casillas
  * prueba final del usuario: el FEN salio EXACTO con los 2 caballos movidos
    (f3/cabs: 'r1bqkb1r/pppppppp/2n5/8/8/5N2/PPPPPPPP/RNBQKB1R') con 32/32.

**Remanente honesto:** entre DOS capturas consecutivas puede fluctuar una
casilla vacia como 'N rojo' (un frame de 2). Con el fine-tune eso BAJA
un poco; la regla `descartar_fantasmas_baja_confianza` (conf < 0.55 y
'vacia' fuerte en el ranking) lo elimina automaticamente en el conector, y
la practica recomendada para el juego en vivo: tomar DOS lecturas y
CONFIRMAR la interseccion de las dos (la herramienta `captura_revision.py`
te imprime las dos para compararlas).

**Causa raiz (documentada):** los sprites 'caballo_n' que alimentaron el
set de compuestos heredaron un glifo cruzado de una sesion anterior (el
usuario coloco el par reversible al inicio); el reentreno real de
la camara arregla el dominio. NO es la luz (probe de CLAHE/gamma/TTA
descartado) y NO es la rotacion del glifo (aplicado TTA rotacion tambien
descartado): los experimentos de descartes quedan en scripts/experimento *
_luz.py y scripts/probe_*.py.

### 8.2 Consideracion fisica del SETUP

1. El tablero debe quedar COMPLETO con margen: las 4 esquinas impresas del
   marco (los codigos ArUco) son lo que rectifica la vista; sin ellas NO hay
   lectura y el conector lanza RuntimeError con motivo.
2. El tablero impreso tiene un sentido fisico: los textos "LADO DE LAS
   NEGRAS" va decidir la orientacion por los codigos (no depende de como
   veas la foto).
3. Las fichas impresas pueden estar giradas a TODO: el clasificador tolera
   rotacion libre del simbolo (fue parte del dataset entrenado).
4. Camara ESP32 (firmware AI-Thinker): sirve `GET /capture` (image/jpeg) a
   640x480; espera >= 0.5 s entre peticiones. El ping ICMP falla por diseño
   del firmware: no lo uses como diagnostico (usa difegetico HTTP: scripts/
   diagnostico_esp32.py --ip <ip>).

### 8.3 Convenciones de salida (el estandar FEN)

* `chess.Board` y la FEN usan el criterio MAYUSCULA = blancas, minuscula =
  negras; 'N' = caballo (porque 'K' ya es el rey), 'B' alfil, 'R' torre,
  'Q' dama, 'K' rey, 'P/p' peon, '.' vacia o el numero en la FEN.
* El tablero retornas SIEMPRE con el turno de las blancas: ajustalo con
  `tablero.turn = chess.BLACK` si tu motor juega negras.

### 8.4 Cosas que NO debes tocar sin avisar

* `classes.txt` y el orden de las 12 clases: los modelos fueron entrenados
  con ESE orden; una clase removida/cambiada rompe el modelo silenciosamente.
* `modelos/`: los pesos que lee el conector por defecto.
* `src/pipeline_fen.py`: es el nucleo compartido de todos los scripts.
* La carpeta `modelos/best_clasificador.pt` es MobileNetV3-Small con la
  HEAD de 13 clases (12 + `vacia`): si reentrenan con otra arquitectura,
  otros usan el mismo contrato pero `cargar_clasificador` debe
  actualizarse en `src/pipeline_fen.py`.

### 8.5 Decisions de la vision: respondidas con datos (docs/d5-notas.md)

* Por que la caja YOLO de los recortes mide 0.85 de casilla y no 0.44:
  se midio en el dataset real (fira: 0.891, debido a propo3cion del print).
* Por que el train del detector es solo sintetico + compuestos y el
  val/test SOLO fotos reales de la camara: regla heredada de v1 cuando doble
  conteo ademas en el mismo dataset vuelve la metrica mas alta con menos
  semilla.
* Por que la Classificator per-casilla y no solo YOLO: el tablero ya es una
  rejilla conocida de 64 casillas (via ArUco); pedirle al YOLO que descubra
  la ubicacion de piezas es resolver un problema que ya tenias resuelto.
  archivo resultados, posicion por posicion: 99.3% contra 91.7% (mAP50 de
  YOLO) para el demo; la historia esta en docs/d5-notas.md y en el
  `docs/ENTREGA.md` (la tabla del capitulo 3).

Contacto directo: revisa `docs/d5-notas.md` (decisiones con los numeros
exactos) si la fecha justifica alguna mecana.

