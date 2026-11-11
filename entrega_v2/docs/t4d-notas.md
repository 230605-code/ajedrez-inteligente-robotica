# T4d — Dataset compuesto con texturas reales

Cubre **S3** (arquitectura v2: datos para el detector YOLO y para el clasificador
por casilla) y **S5** (dataset objetivo del proyecto, aquí ampliado desde las
mismas fotos reales). El sintético de plantilla sigue existiendo en
`dataset/sintetico/`: esto se suma, no lo reemplaza.

## Qué se generó

| Salida | Cantidad |
|---|---|
| `dataset/reales_compuesto/sprites/*.png` | 502 sprites RGBA (12 clases, 15-132 por clase) |
| `dataset/reales_compuesto/images/*.jpg` | 1560 tableros (1092 ESP 640x480 + 468 celular 1024x768) |
| `dataset/reales_compuesto/labels/*.txt` | 1560 archivos YOLO, 44 125 cajas, 0 líneas inválidas |
| `dataset/reales_compuesto/casillas/clasificador/<clase>/*.png` | 128 recortes 100x100 (2 tableros x 64) |
| `dataset/reales_compuesto/manifiesto.csv` | 1560 filas + `homografia_recorte` por imagen |
| `dataset/reales_compuesto/etiquetas_casillas.csv` | 128 filas (imagen, casilla, clase) |
| `dataset/reales_compuesto/rotaciones.csv` | 44 125 filas (ángulo y diámetro por ficha) |
| `assets/previews/compuesto_muestra_1..5.jpg` | 3 frames anotados + 2 hojas de casillas |
| `assets/previews/sprite_muestra.jpg` | 12 sprites al azar |

Reparto: 70 % ESP32 / 30 % celular y 50 % posiciones legales / 50 % aleatorias.
Tiempo medido: 314 s para los 1560 tableros (202 ms/tablero). Tamaño total:
116,9 MB. `n_images == n_labels == 1560`.

## Cómo se separa una ficha del tablero (hallazgo clave)

Umbralizar por brillo no funciona con estas fotos: la ficha **blanca** sobre
casilla **gris** tiene ~20 de contraste y la **negra** sobre casilla **azul**
también se confunde. Lo que sí funcionó fue modelar el tablero vacío:

1. En la vista rectificada cada casilla mide exactamente 100 px.
2. Se ajusta un polinomio cuadrático por canal y por **paridad de casilla**
   (clara/oscura) usando solo anillos de borde de las 32 casillas de esa
   paridad. Predice el color que tendría esa casilla vacía en cada píxel.
3. La distancia en Lab entre la foto y ese modelo es ~6 de mediana en una
   casilla vacía y 22-53 en una ficha. Sobre ese mapa se corre HoughCircles.
4. Todas las fichas de una foto rectificada comparten radio: la mediana de los
   radios detectados es el radio del tablero (**40-44 px** medidos; 41 es el
   valor más común). El centro sale del centroide de la mancha de diferencia.
5. El sprite se recorta con alpha circular, 2 px adentro del radio medido,
   feather de ~2 px. Si el anillo exterior sigue pareciendo tablero (se compara
   contra el fondo predicho, no contra un umbral de saturación), se encoge 1,5 px
   y se repite hasta 4 veces; el número de ajustes queda en el manifiesto.

Sin el paso 2 el recorte arrastraba el damero y los sprites salían con un halo
azul. Con el paso 2, 502 de las 544 casillas iniciales dieron sprite usable: 42
se descartaron por fuga de fondo y 43 tienen el disco cortado por el borde del
recorte (son las fichas de la fila 1/8 y las columnas a/h; se rellenan
replicando el borde, no con negro).

## Decisiones de interpretación (y por qué)

- **Escala 0.92-1.08**: se aplica al **diámetro real del disco** medido en la
  extracción (2 x radio del tablero), no al lado del sprite (que viene 2 px más
  chico por el alpha). Con el diámetro real la ficha ocupa 0,75-0,88 de la
  casilla, que es lo que miden las fotos reales.
- **Sin superposición**: el jitter dentro de la casilla se recorta a
  `0.95 * (50 - radio_disco)` px. Con eso dos fichas vecinas nunca se acercan a
  menos de la suma de sus radios (verificado: ratio mínimo 1,007, 0 violaciones).
- **Armonización de color**: se transfiere **solo la luminancia** del cuadrado de
  origen al de destino. Transferir también el croma a/b teñía los discos de azul
  y borraba el símbolo (se vio en la primera corrida). El balance de blancos
  global lo aplica `_fotometria` a la imagen completa, como una cámara real.
- **Recortes de casilla "a la 01 de cada perfil"**: por defecto solo el primer
  tablero de cada perfil guarda sus 64 recortes (128 archivos). Es literalmente
  lo que pedía el enunciado para no llenar el disco (1560 x 64 serían ~100 000
  PNG). `--casillas-cada N` (o `--sin-casillas`) permite ampliarlo.
- **Perspectiva**: `_colocar` de `generar_sinteticos` con ángulo ±8° y jitter de
  esquinas 3-4 % del lado. Es más fuerte que el jitter del sintético (2,5°/1,2 %
  en ESP).

## Reutilización (no se copió lógica)

- Posiciones legales: `generar_sinteticos._piezas_jugada` (4-40 jugadas desde la
  inicial con `python-chess`).
- Pegado con rotación, fotometría por perfil, cajas YOLO y anotación:
  `_pegar_ficha`, `_fotometria`, `_linea_yolo`, `_anotar`, `_colocar`,
  `matriz_a_texto`, `destino_rectificado` del mismo módulo.
- Posición inicial y lectura del CSV de autoetiquetado:
  `autoetiquetar_reales.INICIAL` (vía `aumentar_reales.leer_etiquetas`).
- Fondo del marco: recortes de esquina de 8 fotos crudas reales (mantel/mesa).

## Limitaciones conocidas

- Las fichas salen más blandas que el tablero porque vienen de fotos ESP de
  640x480 rectificadas de 800 px. Se compensa con un desenfoque leve del tablero
  (sigma 0,3-0,7), pero el perfil ESP es el que menos nitidez tiene.
- Los recortes 100x100 del perfil ESP son celdas de ~55 px ampliadas: es
  exactamente lo que verá el clasificador en producción, así que sirven, pero no
  aportan detalle que la cámara no tenga.
- Quedan 2 PNG huérfanos en `sprites/` (`sprite_alfil_b_027.png`,
  `sprite_dama_b_016.png`) de una corrida anterior con conteos más altos. No
  están en `sprites_manifesto.csv`, así que el compositor los ignora.

## Cómo re-ejecutar

```powershell
$p = "F:\TAVO\Projects\ajedrez-yolo\.venv\Scripts\python.exe"
& $p scripts\extraer_sprites.py                     # 502 sprites + preview
& $p scripts\compilar_compuestos.py --n 1560 --semilla 11   # ~5 min
& $p scripts\compilar_compuestos.py --n 40 --casillas-cada 10  # más recortes
```
