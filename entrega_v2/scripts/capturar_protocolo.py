"""Capturador con el PROTOCOLO del usuario (pausas para ajustar angulo y rotar).

Protocolo por posicion:
  FASE 1 (vista blanca): N capturas cada --segundo 3, con BEEP en T-3, T-2, T-1
    para que el usuario acomode el angulo ligeramente entre fotos.
  PAUSA DE ROTACION: N/4 s con bips largos: el usuario rota el tablero a la
    perspectiva de las negras.
  FASE 2 (vista negra): otras N capturas, mismos bips y mismos segundos.

El conteo continua automaticamente desde el maximo existente en la carpeta
(no sobrescribe). Guarda crudo + registro de calidad (esquinas ArUco) en un
log por carpeta. NO etiqueta: el etiquetado es posterior (FEN declarada).

USO:
    python capturar_protocolo.py --ip 10.236.255.200 --carpeta 01_inicial_intacta \
        --posicion 1 --fase1 8 --fase2 8 --segundo 3 --pausa-rotacion 10
Para despues: cambia --carpeta y --posicion por cada posicion nueva.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

import winsound  # noqa: E402


def bip(alto: bool = False) -> None:
    winsound.Beep(1400 if alto else 800, 180)


def cuenta_regresiva(segundos: int, pie: str) -> None:
    for restante in range(segundos, 0, -1):
        print(f"    T-{restante}s ({pie})", flush=True)
        bip(alto=False)
        time.sleep(1)
    bip(alto=True)
    time.sleep(0.25)
    bip(alto=True)


def capturar(url: str) -> np.ndarray | None:
    import requests

    try:
        respuesta = requests.get(url, timeout=15)
        respuesta.raise_for_status()
        return cv2.imdecode(np.frombuffer(respuesta.content, np.uint8),
                            cv2.IMREAD_COLOR)
    except requests.RequestException as exc:
        print(f"    ERROR {type(exc).__name__}: {str(exc)[:80]}")
        return None


def siguiente_numero(ruta_carpeta: Path) -> int:
    maximo = 0
    for ruta in ruta_carpeta.glob("*.jpg"):
        partes = ruta.stem.rsplit("_", 1)
        if partes[-1].isdigit():
            maximo = max(maximo, int(partes[-1]))
    return maximo + 1


def guardar_con_control(url: str, destino: Path, log) -> bool:
    frame = capturar(url)
    if frame is None:
        log("  guardado FALLIDO (sin frame)")
        return False
    cv2.imwrite(str(destino), frame)
    alta, ancho = frame.shape[:2]
    control = "OK"
    try:
        from esquinas_aruco import detectar_robusto
        esquinas, metodo, detalle = detectar_robusto(frame)
        control = f"{metodo} ({str(detalle)[:60]})" if esquinas is not None \
            else f"sin esquinas ({str(detalle)[:50]}) sin control total"
    except Exception as exc:  # noqa: BLE001
        control = f"control de esquinas fallo: {exc}"
    log(f"  guardado {destino.name}  {ancho}x{alta}  {len(destino.name) and ''}"
        f"[{control}]")
    return frame.shape[0] >= 480


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="10.236.255.200")
    parser.add_argument("--url")
    parser.add_argument("--carpeta", required=True,
                        choices=("01_inicial_intacta", "02_inicial_con_jugadas",
                                 "03_medio_juego", "04_finales",
                                 "05_tablero_vacio"))
    parser.add_argument("--posicion", type=int, required=True,
                        help="numero de posicion fisica (1..N) para nombrar")
    parser.add_argument("--fase1", type=int, default=8)
    parser.add_argument("--fase2", type=int, default=8)
    parser.add_argument("--segundo", type=float, default=3.0)
    parser.add_argument("--pausa-rotacion", type=int, default=10)
    parser.add_argument("--sin-fase2", action="store_true",
                        help="omitir la fase 2 (vista de negras)")
    parser.add_argument("--fen", default="",
                        help="la FEN DECLARADA de la posicion (para el manifesto; "
                             "en 01_inicial_intacta se usa la del estandar auto)")
    args = parser.parse_args()

    url = args.url or f"http://{args.ip}/capture"
    carpeta = RAIZ / "dataset" / "crudo" / args.carpeta
    carpeta.mkdir(parents=True, exist_ok=True)
    log_path = carpeta / "capturas.log"

    def log(mensaje: str) -> None:
        print(mensaje, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(mensaje + "\n")

    log(f"\n=== POSICION {args.posicion} de {args.carpeta}  "
        f"(fase1={args.fase1}, fase2={args.fase2}) ===")

    if args.carpeta == "01_inicial_intacta":
        fen_para_etiqueta = "inicial_estandar"
    elif args.fen:
        fen_para_etiqueta = args.fen
    else:
        fen_para_etiqueta = "(SIN DECLARAR)"

    # --- MANIFESTO: archivo -> posicion -> FEN ---
    manifiesto = carpeta / "manifesto.csv"
    registered: dict[str, str] = {}
    if manifiesto.exists():
        import csv
        for fila_csv in csv.DictReader(
                manifiesto.open(encoding="utf-8-sig", newline="")):
            registered[fila_csv["archivo"]] = fila_csv["posicion"]

    def registrar(archivo: str, fuente: str) -> None:
        import csv
        existe = manifiesto.exists()
        alta_total = archivo not in registered
        if alta_total or fuente == "auto":
            estados_old = []
            if existe:
                with manifiesto.open(encoding="utf-8-sig", newline="") as f:
                    estados_old.extend(csv.DictReader(f))
            nueva = [{"archivo": registro["archivo"],
                      "posicion": registro["posicion"],
                      "fen": registro["fen"],
                      "fuente": registro["fuente"]}
                     for registro in ([{"archivo": archivo,
                                        "posicion": args.posicion,
                                        "fen": fen_para_etiqueta,
                                        "fuente": fuente}]
                                      + [r for r in estados_old
                                         if r["archivo"] != archivo])]
            with manifiesto.open("w", encoding="utf-8-sig", newline="") as f:
                escritor = csv.DictWriter(f, fieldnames=["archivo", "posicion",
                                                        "fen", "fuente"])
                escritor.writeheader()
                escritor.writerows(nueva)
            registered[archivo] = str(args.posicion)

    # Los archivos ya en el carpeta SIN registro = subidas manuales del
    # usuario para ESTA posicion (por el protocolo acordado).
    manuales = [r.name for r in carpeta.glob("*.jpg")
                if r.name not in registered and r.name != "capturas.log"]
    if manuales:
        for nombre in(manuales):
            registrar(nombre, "usuario")
        log(f"  manifesto: {len(manuales)} fotos manuales del usuario "
            "regadas a la posicion " + str(args.posicion))

    contador_uf = siguiente_numero(carpeta)
    sufijo = f"p0{args.posicion}" if args.posicion < 10 else f"p{args.posicion}"

    # camera viva?
    prueba = capturar(url)
    if prueba is None:
        print("la camara no responde; cancela")
        return 1
    log(f"  camara OK: {prueba.shape[1]}x{prueba.shape[0]}")

    tomadas_ok = 0
    mis_archivos = []
    print("\n--- FASE 1 (vista blanca/actual) ---", flush=True)
    for indice in range(args.fase1):
        if indice:
            cuenta_regresiva(int(args.segundo), "cambia ligeramente el angulo")
        objetivo = carpeta / (f"{args.carpeta}_{sufijo}_{contador_uf + indice:03d}.jpg")
        if guardar_con_control(url, objetivo, log):
            tomadas_ok += 1
            mis_archivos.append(objetivo.name)

    # El contador avanza con los archivos reales en disco.
    contador_uf = siguiente_numero(carpeta)

    if not args.sin_fase2:
        print(f"\n--- PAUSA DE ROTACION ({args.pausa_rotacion}s): "
              "ROTA EL TABLERO A LA PERSPECTIVA DE LAS NEGRAS ---", flush=True)
        cuenta_regresiva(args.pausa_rotacion, "rota el tablero")
        print("\n--- FASE 2 (vista negras) ---", flush=True)
        for indice in range(args.fase2):
            if indice:
                cuenta_regresiva(int(args.segundo), "cambia ligeramente el angulo")
            objetivo = carpeta / (
                f"{args.carpeta}_{sufijo}_{contador_uf + indice:03d}.jpg")
            if guardar_con_control(url, objetivo, log):
                tomadas_ok += 1
                mis_archivos.append(objetivo.name)

    log(f"\n=== POSICION {args.posicion} terminada: {tomadas_ok} fotos "
        f"(files total en carpeta: {len(list(carpeta.glob('*.jpg')))}) ===\n")
    for indice in range(args.fase1):
        pass  # el registro 'auto' de estas capturas se hace al final
    for nombre_archivo in mis_archivos:
        registrar(nombre_archivo, "auto")
    log("manifesto: " + str(len(mis_archivos)) + " fotos auto registradas ~ posicion " + str(args.posicion))

    print("\nListo: cuando montes la SIGUIENTE posicion me escribes 'listo'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
