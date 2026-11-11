# ajedrez-v2 — Detector de piezas con fichas nuevas, cámara mejorada y clasificador por casilla

Proyecto nuevo desde cero, aprovechando las lecciones de `ajedrez-yolo` (v1).
No se toca v1: queda como referencia y línea base para comparar.

## Qué cambió respecto a v1

| Cambio | Efecto |
|---|---|
| Fichas nuevas con base negra y blanca | El color dejó de depender del relleno del símbolo: lo distingue el propio cuerpo de la ficha |
| Símbolos más entendibles y detectables | Sube la clasificación de forma (el error confirmado en v1) |
| Cámara ESP32 mejorada para capturas | Más resolución o mejor exposición → piezas más legibles |
| Arquitectura v2 | Detección YOLO (línea base) **+ clasificador por casilla sobre el tablero rectificado** + post-filtro de movimientos legales |

## Estructura

```
ajedrez-v2/
├── README.md
├── classes.txt               Orden oficial de las 12 clases (igual que v1)
├── dataset/
│   ├── crudo/
│   │   ├── 01_inicial/        Posición estándar
│   │   ├── 02_medio_juego/    12-20 piezas en posiciones variadas
│   │   ├── 03_finales/        4-8 piezas (muchas casillas vacías)
│   │   ├── 04_dificiles/      Reinas/reyes solos, bordes, contrastes raros
│   │   └── 05_tablero_vacio/  Sin piezas: control de falsos positivos
│   └── etiquetado/            Export de Roboflow (images/ + labels/)
├── modelos/                   modelo_sintetico.pt heredado, best.pt al terminar
├── scripts/                   Captura, diagnóstico, preparación, evaluación
├── src/                       Pipeline v2
├── assets/previews/           Capturas y vistas anotadas para revisión
├── resultados/                Métricas, gráficas y corridas de entrenamiento
└── docs/                      Guias y protocolos
```

## Protocolo de captura de imágenes (OBLIGATORIO)

1. **Cámara fija.** Misma altura y distancia para todo el dataset. Esa constancia
   es lo que permite calibrar las esquinas UNA vez por sesión y derivar
   automáticamente los 64 recortes de casilla del clasificador.
2. **Tablero completo en el encuadre, con margen.** Las 4 esquinas visibles.
   Si se corta una esquina, la foto se descarta: sin esquina no hay homografía.
3. **Resolución máxima estable** de la cámara. Verificar con
   `scripts/diagnostico_esp32.py` antes de capturar en serio.
4. **Dos sesiones de luz**: día y noche con lámpara (aprox. 60% / 40% de las
   fotos). Con una sola iluminación el modelo aprende la luz, no las piezas.
5. **No editar nada**: sin recortes, sin renombres, sin retocar brillo o color.
   Subir tal cual llega de `/capture`, en JPG.
6. **Nombrado**: los scripts generan `01_inicial_001.jpg`, `01_inicial_002.jpg`,
   ... por carpeta y en orden de captura. No renombrar a mano.
7. **Entre foto y foto**: mover 1 a 3 piezas para variar la posición. En
   finales y difíciles, montar la posición antes de capturar.

## Objetivo de imágenes (dataset completo)

| Carpeta | Fotos | Nota |
|---|---|---|
| 01_inicial | 40 | 32 fichas, cámara fija |
| 02_medio_juego | 40 | 12-20 fichas, posiciones variadas |
| 03_finales | 30 | 4-8 fichas (alimenta la clase "vacía") |
| 04_dificiles | 20 | Reinas y reyes aislados, bordes, piezas sobre casilla de su mismo color |
| 05_tablero_vacio | 10 | Sin fichas, tablero limpio |
| **Total** | **140** | |

Requisito de sesión: **las 140 fotos se reparten entre las 2 sesiones de luz**.
Mejor 70 fotos para tope que 140 de un solo tipo de iluminación.

## Modelo v2: cuál es la diferencia con YOLO

YOLO detecta cajas e ignora el hecho de que el tablero es una rejilla conocida
de 64 casillas. v2 lo aprovecha:

1. Se rectifica el tablero con la homografía (esquinas fijas por sesión).
2. Se recortan las 64 casillas a tamaño fijo.
3. Un clasificador liviano (MobileNetV3 o EfficientNet-B0) decide entre
   **13 clases**: las 12 piezas + `vacia`.
4. Se aplican movimientos legales con `python-chess` como post-filtro.

Las 140 fotos sirven para AMBOS modelos: YOLO como línea base comparable y el
clasificador como propuesta v2, con los recortes generados automáticamente desde
las etiquetas. Sin etiquetado extra.

Estado actual: **paso 0** — validar la cámara y las fichas nuevas con 8 capturas
de la posición inicial antes de capturar las 140.

## Reglas que no cambian

- Todo el trabajo en `F:\TAVO\Projects`, nunca en `C:`.
- La llave de Roboflow va SOLO en la variable de entorno `ROBOFLOW_API_KEY`.
- El orden de `classes.txt` es el mismo de v1 y no se toca.
- Entrenamiento en Google Colab sobre el notebook adaptado.

---

## Árbitro de Ajedrez Inteligente (Aplicación de Escritorio)

La interfaz gráfica de escritorio unifica la visión de la ESP32-CAM, las reglas de ajedrez, el motor Stockfish y el control de un brazo robótico.

### Cómo ejecutar la aplicación

Doble clic en:
```cmd
iniciar_interfaz.bat
```
O directamente desde la consola con el entorno virtual del proyecto:
```powershell
.\.venv\Scripts\python.exe iniciar_interfaz.py
```
*(Opcional: `--ip <DIRECCION_IP>` para cambiar la IP de la cámara o `--offline <RUTA_IMAGEN>` para pruebas sin cámara).*

### Características principales

1. **Detección Automática de Movimientos (Manos Libres):**
   - El sensor mide la diferencia de cuadros para detectar cuándo la mano entra al tablero.
   - Espera a que el tablero quede completamente en reposo (1.3 s de estabilidad) para capturar y deducir la jugada automáticamente sin intervención manual.
2. **Validación de Reglas y Explicación Pedagógica en Español:**
   - Si se detecta un movimiento ilegal (turno incorrecto, pieza clavada, rey en jaque, casilla ocupada), el árbitro bloquea la partida, explica la infracción y detalla cómo devolver físicamente la pieza a su casilla original.
   - En cuanto el jugador devuelve la pieza, el sistema detecta la corrección y reanuda el juego.
3. **Análisis Continuo con Stockfish:**
   - Barra vertical de evaluación posicional y porcentaje de probabilidad de victoria.
   - Flecha gráfica sobre el tablero vectorizado señalando la mejor respuesta recomendada por el motor.
4. **Despacho de Eventos para Brazo Robótico:**
   - Genera objetos `EventoJugadaRobot` (UCI, SAN, origen, destino, coordenadas matriciales 0..7, tipo de pieza, color, captura, enroque, promoción) listos para comunicarse por puerto serie / socket con un microcontrolador físico.
5. **Modos de Inicio:**
   - Nueva partida estándar.
   - Escanear tablero físico actual (deduce FEN desde la cámara).
   - Cargar FEN arbitraria.
   - Cambio manual de turno (blancas/negras).

