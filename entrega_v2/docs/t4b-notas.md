# T4b — Esquinas robustas, recorte de reales y dataset del clasificador

Notas de la tarea T4b. Cubren **S2** (las fotos reales con la camara y las fichas
nuevas se pueden usar de verdad: el tablero se detecta en el 95.5% de las tomas)
y **S3** (arquitectura v2: queda listo el dataset por casilla del clasificador y
el aumento que lo alimenta).

Antes de esta tarea se recortaban **25 de 134** fotos reales (18.7%): el detector
ArUco exigia los 4 marcadores y en las fotos de ESP32 suele faltar uno. Ahora se
recortan **128 de 134** (95.5%) y cada recorte pasa una validacion geometrica
independiente del detector que lo produjo.

## 1. Las tres estrategias (`scripts/esquinas_aruco.py`)

`detectar_robusto(imagen) -> (esquinas, metodo, detalle)` prueba en orden y
devuelve el **primer** resultado que pasa la validacion comun.

| Metodo | Como encuentra el tablero | Cuando gana |
|---|---|---|
| `aruco-multi-escala` | ArUco a 1.0x, 2.0x, 2.5x y 3.0x sobre gris, fusionado por id; homografia con las 4 esquinas de CADA marcador | 123 de 128 |
| `cuadriculado` | centros de las 32 casillas azules del tablero impreso, ajustados a la reticula conocida (o `findChessboardCornersSB` sobre la mascara) | 2 de 128 |
| `piezas` | rejilla inferida de las detecciones del `best.pt` de v1 (`comun.esquinas_por_rejilla`) | 3 de 128 |
| `ninguno` | ninguna paso la validacion | 6 de 134 |

Claves de implementacion:

- **Con 2 marcadores ya hay homografia.** Antes se exigian 4. Cada marcador
  aporta 4 esquinas exactas, asi que 2 dan 8 correspondencias. Es lo que
  convierte un "3/4 marcadores" en un recorte valido.
- **Fusion multi-escala ponderada por area.** La deteccion mas grande tiene mas
  pixeles por lado y esquinas mas finas. Escalas que superarian 60 Mpx se
  omiten (la 3.0x sobre una foto de 4080x2296 serian 84 Mpx).
- **Orden canonico, no orden de imagen.** `detectar()` reordenaba las esquinas
  por su posicion en la imagen (`ordenar_esquinas`), lo que **gira 90 grados**
  el recorte de una foto tomada de lado. `detectar_robusto` conserva el orden de
  la plantilla (TL = lado de las negras), asi que la vista sale siempre con las
  blancas abajo venga como venga la camara.
- **Bug corregido.** El bucle de residuales de `detectar()` recorria
  `pares` como `(id_plantilla, id_detectado)` cuando `_emparejar` devuelve
  `(id_detectado, id_plantilla)`. Con los ids iguales el error era invisible;
  con el emparejado por lado daba un residual falso. Ahora esta en el orden
  correcto y salta los pares incompletos en vez de lanzar `KeyError`.

## 2. La validacion: el damero impreso

Las tres estrategias se validan con lo mismo: **¿la rejilla rectificada acierta
el color impreso de las 64 casillas?** Se mide el canal `b*` de LAB en las
esquinas de cada casilla (el centro lo tapa la ficha) y se compara la diferencia
entre casillas vecinas con el signo que impone el patron azul/gris.

Es un filtro adaptado local, no un umbral de color global. Medido sobre las 134
fotos:

| Situacion | Concordancia |
|---|---|
| Homografia correcta | 0.85 a 1.00 |
| Corrida una casilla | -1.00 |
| Corrida media casilla | ~0.00 |
| Corrida un tercio (2 marcadores del mismo lado) | 0.66 a 0.70 |

Umbral de aceptacion: **0.80**. De los 128 recortes aceptados, la concordancia
minima es 0.8457 y la mediana 1.00. Se probaron dos alternativas antes:

- umbral fijo de saturacion HSV: 50% en las fotos nocturnas (la mitad del
  tablero en sombra se lee como gris);
- k-medias global de `b*` sobre las 64 casillas: 70-84% cuando una esquina del
  tablero cae en sombra, y no distingue una casilla corrida de una ficha.

Comparar vecinos cancela la iluminacion (que cambia despacio) y conserva el
patron (que cambia celda a celda).

## 3. Resultado del recorte

```powershell
python scripts/recortar_reales.py --metodo robusto
```

| Dato | Antes | Ahora |
|---|---|---|
| Fotos recortadas | 25 / 134 (18.7%) | **128 / 134 (95.5%)** |
| Desbordes (pixeles de fuera) | 0 | **0** (max 0.0000%) |
| Tiempo | — | 79.2 s |

Reparto: `aruco-multi-escala` 123, `cuadriculado` 2, `piezas` 3. Por carpeta:
83 de `01_inicial`, 37 de `02_medio_juego`, 5 de `03_finales`, 3 de
`05_tablero_vacio`.

Las 6 que quedan fuera son todas fotos de ESP32 de 640x480 muy oscuras, con 0 o
1 marcador util y menos de 13 casillas azules detectables. El CSV de auditoria
(`dataset/auditoria/recortes_reales.csv`) lleva por foto: estado, metodo, si el
recorte es entero, los ids vistos, la confianza (concordancia de color), la
orientacion usada, los inliers de la reticula, el rms y el motivo.

`--metodo aruco` conserva el camino viejo (solo ArUco, exige 4/4) para comparar.

## 4. Hallazgos sobre la hoja impresa

1. **El marco impreso no coincide exactamente con `plantilla/tablero.png`.**
   En la hoja fisica los marcadores quedan practicamente sobre las esquinas del
   area jugable; en la plantilla estan 0.33 casillas mas afuera. La homografia
   de 4 marcadores sale entonces 2-6% mas grande que el tablero real. No rompe
   el recorte (la validacion de color lo tolera) pero explica por que el
   cuadriculado del propio tablero da esquinas mas finas.
2. **Los textos "LADO DE LAS NEGRAS/BLANCAS" van girados en la hoja fisica**
   respecto de la plantilla. No afecta a la geometria: el patron azul/gris y los
   ids si coinciden, y la vista canonica sale con las negras arriba.
3. **Convencion de colores confirmada con las imagenes:** disco NEGRO con
   simbolo blanco = `_b` (blanca); disco BLANCO con simbolo negro = `_n`
   (negra). En la posicion inicial las blancas van abajo (filas 6-7 de la vista)
   y las negras arriba.
4. **`01_inicial` no contiene la posicion inicial en la mayoria de las fotos.**
   Solo 17 de los 83 recortes de esa carpeta coinciden exactamente con las 32
   fichas iniciales. El resto son medio juego: muchas con exactamente 5 piezas
   movidas, que es lo que pide el protocolo de captura ("mover 1 a 3 piezas entre
   foto y foto"). No es un fallo del detector: se verifico a ojo en varias.

## 5. Autoetiquetado (`scripts/autoetiquetar_reales.py`)

Decide ocupada/vacia por **contraste dentro de la casilla** (se recorta un 8% de
margen para no medir la linea entre casillas).

La medida original era la std del gradiente. En las fotos de 640x480, blandas de
foco, el gradiente se aplana: en `capture (11).jpg` las fichas daban 13-50 y las
vacias 3-8, con el corte de k-medias en 17 (dos fichas perdidas). Se cambio a la
**std de intensidad**, que mide lo mismo que importa (una casilla vacia es una
superficie plana) y conserva la separacion: fichas 13-66, vacias 1-5. El
gradiente se sigue midiendo para el detalle del CSV.

El corte es el **punto medio del mayor hueco** entre los 64 valores ordenados.
k-medias de dos grupos no sirve: con 32 fichas y 32 vacias los grupos pesan
igual y el centro alto cae en medio de la dispersion de las fichas, cortando las
mas apagadas. Un tablero vacio no tiene hueco grande ni nivel alto, y da 0
ocupadas.

Resultado:

| Grupo | Recortes |
|---|---|
| Autoetiquetados (posicion inicial exacta) | 17 |
| Autoetiquetados (tablero vacio) | 3 |
| `desconocida` (ocupacion distinta de la inicial) | 66 |
| `manual` (02_medio_juego, 03_finales) | 42 |

Ademas se comprueba el color: si los discos claros no quedan arriba, el recorte
esta girado 180 grados y se descarta en vez de etiquetar el color al reves.

El CSV `dataset/reales/etiquetas_clasificador.csv` tiene
`archivo_recorte, casilla, clase, fila, estado, motivo`: 64 filas por recorte
etiquetable y UNA fila (`estado=desconocida|manual`) para el resto.

## 6. Aumento (`scripts/aumentar_reales.py`)

Solo aumenta los recortes con `estado=auto`. 6 variantes deterministas por
recorte (semilla por `crc32` del nombre, no `hash()`, que cambia entre procesos):
brillo/contraste 0.75-1.35, gamma 0.7-1.4, saturacion 0.8-1.2, ruido sigma 3-12,
desenfoque sigma 0-1.5, rotacion +-5 grados y jitter de perspectiva de hasta 2%.

El jitter **no desalinea las casillas**: se deforma la imagen completa y las 64
casillas se recortan con la MISMA homografia, asi que la etiqueta viaja con la
geometria. Un jitter que desplazara la rejilla seria basura bien etiquetada.

```powershell
python scripts/aumentar_reales.py
```

| Salida | Contenido |
|---|---|
| `dataset/reales/augmentado/images/` | 120 imagenes (20 recortes x 6) |
| `dataset/reales/augmentado/etiquetas_clasificador.csv` | 7680 filas (mas columna `variante`) |
| `dataset/reales/clasificador/casillas/<clase>/` | 7680 recortes de 100x100 en 13 carpetas |
| `assets/previews/augmentado_muestra_1..4.jpg` | hojas de contacto (6 variantes) |

Por clase de casilla: `peon_b` 816, `peon_n` 816, `vacia` 4416, `torre_*` 204,
`caballo_*` 204, `alfil_*` 204, `dama_*` 102, `rey_*` 102. Ninguna en 0.

## 7. Limitaciones y siguientes pasos

- **Faltan posiciones reales variadas etiquetadas.** 17 recortes iniciales + 3
  vacios dan 1280 casillas reales; el resto de `01_inicial` es medio juego sin
  posicion conocida. Para el clasificador conviene etiquetar a mano una muestra
  de `02_medio_juego` (el CSV ya marca cuales) o completar la captura de
  `04_dificiles` (0 de 20) y `03_finales` (5 de 30).
- **6 fotos siguen sin recorte**, todas de ESP32 640x480 muy oscuras. Se
  resolverian con una toma mejor, no con mas codigo.
- **El camino `piezas` es el mas fragil** (modelo v1 con las fichas viejas),
  pero sus 3 casos dan concordancia de color 0.93-0.99.
- `detectar()` (el camino viejo) sigue reordenando las esquinas por posicion en
  la imagen. No se toco para no cambiar los recortes sinteticos ya generados;
  todo lo nuevo usa `detectar_robusto`.

## 8. Pruebas

No hay runner de tests en el proyecto (ni `tests/`, ni `pytest`), y la
superficie permitida de esta tarea no incluye crear uno. La verificacion es la
corrida sobre las 134 fotos con umbrales medidos y los conteos de arriba, mas
una revision visual de las previews de `assets/previews` y
`dataset/auditoria/previews_autolabel`.
