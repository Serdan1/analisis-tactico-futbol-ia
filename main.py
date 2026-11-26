"""Ejecuta el sistema completo sobre un clip (tarea #15).

    python main.py --video data/clips/2e57b9_0.mp4 --track_id 7
    python main.py --video data/clips/2e57b9_0.mp4 --clic          # elegir al jugador con el ratón

Fases (cada una guarda su resultado en outputs/<clip>/ y la siguiente lo lee):

    1. video       vídeo -> posiciones_todos.csv + homografia.csv     (YOLO + ByteTrack + homografía, GPU)
    2. huellas     -> embeddings.npy                                  (ResNet-18 + color, GPU)
    3. equipos     -> equipos.csv + equipos.json                      (K-Means + KNN, sentido de ataque)
    4. jugador     -> posiciones.csv                                  (re-identificación con galería y tau)
    5. metricas    -> metricas.json + figuras                         (Alex)
    6. informe     -> informe.md                                      (Pablo)

Con --desde se repite a partir de una fase reutilizando lo ya guardado, p. ej.
--desde jugador para probar otro --track_id sin volver a pasar YOLO.
Las fases 5 y 6 se saltan (con aviso) mientras no estén programadas.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

FASES = ["video", "huellas", "equipos", "jugador", "metricas", "informe"]
TAU_POR_DEFECTO = 0.80        # calibrado con FAR/FRR en 2e57b9_0 (notebook 09)
PESOS_RESNET = Path("models/resnet18-f37072fd.pth")

# supervision avisa de nombres que cambiarán en su v1.0: no es un fallo nuestro
warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")


class ErrorPipeline(Exception):
    """Fallo con un mensaje pensado para el usuario (sin traza de Python)."""


def paso(n, texto):
    print(f"\n[{n}/{len(FASES)}] {texto}", flush=True)


def necesita(ruta: Path, fase: str):
    if not ruta.exists():
        raise ErrorPipeline(f"Falta {ruta}. Ejecuta desde la fase '{fase}' (--desde {fase}).")


def medidas_del_campo(a, salida) -> tuple[float, float]:
    """Medidas con las que se calcularon las posiciones (campo.json), o las de --campo si aún no hay.

    Las fases posteriores a 'video' tienen que usar las mismas medidas que la homografía.
    """
    ruta = salida / "campo.json"
    if ruta.exists():
        c = json.load(open(ruta, encoding="utf-8"))
        guardado = (c["largo_m"], c["ancho_m"])
        if a.campo_medidas != guardado and a.campo is not None:
            print(f"  ⚠ Las posiciones se calcularon con un campo de {guardado[0]:g} x {guardado[1]:g} m; "
                  f"se ignora --campo {a.campo}. Para cambiarlo: --desde video")
        return guardado
    if a.campo is not None:   # resultados antiguos, de antes de existir --campo
        print(f"  ⚠ Las posiciones guardadas se calcularon con 105 x 68 m; se ignora --campo {a.campo}. "
              "Para cambiarlo: --desde video")
    return 105.0, 68.0


# --------------------------------------------------------------- fases
def fase_video(a, salida):
    from src.posiciones import MODELO_CAMPO, procesar_video
    print(f"  Modelo de puntos del campo: {a.modelo_campo or MODELO_CAMPO}")
    largo, ancho = a.campo_medidas
    print(f"  Campo de {largo:g} x {ancho:g} m" + ("" if a.campo else " (por defecto; cámbialo con --campo LARGOxANCHO)"))
    todos, diag = procesar_video(a.video, device=a.device, progreso=not a.silencio,
                                 modelo_campo=a.modelo_campo, min_inliers=a.min_inliers,
                                 largo_m=largo, ancho_m=ancho)
    if todos.empty:
        raise ErrorPipeline("No se ha detectado ningún jugador en el vídeo.")
    todos.to_csv(salida / "posiciones_todos.csv", index=False)
    diag.to_csv(salida / "homografia.csv", index=False)
    json.dump({"largo_m": largo, "ancho_m": ancho}, open(salida / "campo.json", "w", encoding="utf-8"))
    pct_h = diag.valida.mean()
    print(f"  Homografía válida en el {pct_h:.0%} de los fotogramas · {todos.track_id.nunique()} tracks")
    if pct_h < 0.5:
        print("  ⚠ En más de la mitad del clip no se ven 4 puntos del campo fiables: "
              "las posiciones serán parciales (¿plano corto o repetición?).")
    return dict(homografia_valida_pct=round(100 * pct_h, 1), tracks=int(todos.track_id.nunique()),
                campo_m=[largo, ancho])


def fase_huellas(a, salida):
    from src.fase1_tracking import ExtractorApariencia, calcular_embeddings
    necesita(salida / "posiciones_todos.csv", "video")
    todos = pd.read_csv(salida / "posiciones_todos.csv")
    extractor = ExtractorApariencia(a.device, pesos=str(PESOS_RESNET) if PESOS_RESNET.exists() else None)
    embs = calcular_embeddings(a.video, todos, extractor, progreso=not a.silencio)
    np.save(salida / "embeddings.npy", embs)
    return dict(huellas=int(len(embs)))


def fase_equipos(a, salida):
    from src.fase1_equipos import equipos_automaticos
    necesita(salida / "embeddings.npy", "huellas")
    todos = pd.read_csv(salida / "posiciones_todos.csv")
    embs = np.load(salida / "embeddings.npy")
    eq, sentido, _ = equipos_automaticos(a.video, todos, embs, largo_m=medidas_del_campo(a, salida)[0])
    eq[["frame", "track_id", "clase", "equipo", "es_portero", "prob"]].to_csv(salida / "equipos.csv", index=False)
    json.dump({"sentido": sentido}, open(salida / "equipos.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    for e, s in sentido.items():
        print(f"  {e:7s} ataca hacia la {s['ataca']} de la cámara (según {s['metodo']})")
    if not sentido:
        print("  ⚠ No se ha podido saber el sentido de ataque: usa --ataca_izquierda o --ataca_derecha.")
    return dict(sentido=sentido)


def fase_jugador(a, salida):
    from src.fase1_equipos import equipo_de_un_jugador
    from src.fase1_tracking import elegir_jugador_con_clic, seguir_jugador
    necesita(salida / "equipos.csv", "equipos")
    todos = pd.read_csv(salida / "posiciones_todos.csv")
    diag = pd.read_csv(salida / "homografia.csv")
    embs = np.load(salida / "embeddings.npy")
    eq = todos.merge(pd.read_csv(salida / "equipos.csv")[["frame", "track_id", "equipo", "es_portero"]],
                     on=["frame", "track_id"], how="left")
    sentido = json.load(open(salida / "equipos.json", encoding="utf-8"))["sentido"]
    LARGO_M, ANCHO_M = medidas_del_campo(a, salida)

    # 1. ¿Quién?
    track_id = a.track_id
    if track_id is None:
        if not a.clic:
            raise ErrorPipeline("Di a qué jugador seguir: --track_id N o --clic.")
        try:
            track_id = elegir_jugador_con_clic(a.video, todos, a.frame_inicio)
        except ValueError:
            raise ErrorPipeline("No has elegido ningún jugador. Lo anterior está guardado: repite solo este paso con\n"
                                f"  python main.py --video {a.video} --clic --desde jugador")
    propio = todos[(todos.track_id == track_id) & (todos.frame >= (a.frame_inicio or 0))]
    print(f"  Jugador elegido: {track_id}")
    if propio.empty:
        primeros = todos[todos.frame == todos.frame.min()].track_id.tolist()
        raise ErrorPipeline(f"El jugador {track_id} no aparece en el clip. "
                            f"En el primer fotograma están: {primeros}. Prueba con --clic.")

    # 2. Seguirlo con la galería (re-identificación, #9)
    seg = seguir_jugador(todos, embs, track_id, a.tau, frame_inicio=a.frame_inicio)
    cambios = seg[seg.cambio_id == 1]
    for r in cambios.itertuples():
        print(f"  fotograma {r.frame}: el tracker le cambió el número → ahora es el {r.track_id}")

    # 3. Equipo y sentido de ataque
    equipo = equipo_de_un_jugador(eq, propio.head(50))
    if a.ataca_izquierda or a.ataca_derecha:
        ataca_izq, origen = a.ataca_izquierda, "a mano"
    elif equipo in sentido:
        ataca_izq, origen = sentido[equipo]["ataca"] == "izquierda", sentido[equipo]["metodo"]
    else:
        ataca_izq, origen = False, "desconocido"
        print("  ⚠ Sentido de ataque desconocido: se dejan las coordenadas de la cámara.")

    # 4. posiciones.csv (formato de docs/formatos.md)
    pos = diag[["frame", "t_s"]].merge(seg[["frame", "x_m", "y_m", "valido"]], on="frame", how="left")
    pos["valido"] = pos.valido.fillna(0).astype(int)
    pos.insert(2, "track_id", track_id)
    pos.insert(3, "equipo", equipo)
    if ataca_izq:
        pos["x_m"] = (LARGO_M - pos.x_m).round(2)
        pos["y_m"] = (ANCHO_M - pos.y_m).round(2)
    pos.to_csv(salida / "posiciones.csv", index=False)
    pct = pos.valido.mean()
    lado = f"izquierda (girado para que ataque hacia x = {LARGO_M:g})" if ataca_izq else "derecha"
    print(f"  Jugador {track_id} ({equipo or 'equipo ?'}) localizado en el {pct:.1%} de los fotogramas · "
          f"ataca hacia la {lado} · sentido según: {origen}")
    if pct < 0.5:
        print("  ⚠ Se le ve en menos de la mitad del clip: las métricas serán poco fiables.")
    return dict(jugador={"id": int(track_id), "equipo": equipo, "rol": a.rol},
                ataca_izquierda=bool(ataca_izq), origen_sentido=origen,
                frames_validos_pct=round(100 * pct, 1), cambios_de_numero=int(len(cambios)))


def fase_metricas(a, salida, resumen):
    from src import fase2_metricas, visualizaciones
    necesita(salida / "posiciones.csv", "jugador")
    jugador = resumen.get("jugador") or {"id": a.track_id, "equipo": "", "rol": a.rol}
    try:
        metricas = fase2_metricas.metricas_completas(salida / "posiciones.csv", jugador, clip=Path(a.video).name)
    except NotImplementedError:
        print("  (pendiente: fase2_metricas.metricas_completas todavía no está programada — Alex, #7/#8)")
        return dict(metricas="pendiente")
    except ValueError as e:      # p. ej. "No hay posiciones validas": el jugador casi no se ve
        raise ErrorPipeline(f"No se pueden calcular las métricas: {e}. "
                            "¿Se ve al jugador en el clip? Prueba con otro --track_id o --clic.")
    json.dump(metricas, open(salida / "metricas.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    try:
        figs = visualizaciones.figuras_completas(salida / "posiciones.csv", salida / "figuras")
        print(f"  {len(figs)} figuras en {salida / 'figuras'}")
    except NotImplementedError:
        print("  (pendiente: visualizaciones.figuras_completas — Alex)")
    except ValueError as e:
        print(f"  ⚠ No se han podido hacer todas las figuras: {e}")
    print(f"  metricas.json guardado")
    return dict(metricas="ok")


def fase_informe(a, salida):
    from src import fase3_informe
    if not (salida / "metricas.json").exists():
        print("  (sin metricas.json: no se puede generar el informe todavía)")
        return dict(informe="pendiente")
    metricas = json.load(open(salida / "metricas.json", encoding="utf-8"))
    try:
        texto, sin_verificar = fase3_informe.informe_completo(metricas)
    except NotImplementedError:
        print("  (pendiente: fase3_informe.informe_completo todavía no está programada — Pablo)")
        return dict(informe="pendiente")
    except Exception as e:      # LLM caído, sin clave, sin conexión...
        try:
            (salida / "prompt.txt").write_text(fase3_informe.construir_prompt(metricas), encoding="utf-8")
            extra = f" Se ha guardado el prompt en {salida / 'prompt.txt'} para probarlo a mano."
        except Exception:
            extra = ""
        print(f"  ⚠ El LLM no ha respondido ({type(e).__name__}: {e}). Las métricas sí están guardadas.{extra}")
        return dict(informe="error", error_llm=str(e))
    (salida / "informe.md").write_text(texto, encoding="utf-8")
    if sin_verificar:
        print(f"  ⚠ Cifras del informe que no están en metricas.json: {sin_verificar}")
    print(f"  Informe guardado en {salida / 'informe.md'}")
    return dict(informe="ok", cifras_sin_verificar=list(map(str, sin_verificar)))


# --------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Análisis táctico de un jugador a partir de vídeo")
    ap.add_argument("--video", required=True, help="ruta RELATIVA al clip .mp4 (OpenCV falla con tildes en la ruta)")
    quien = ap.add_mutually_exclusive_group()
    quien.add_argument("--track_id", type=int, help="número del jugador a seguir (el que sale en control.mp4)")
    quien.add_argument("--clic", action="store_true", help="elegir al jugador haciendo clic en el vídeo")
    ap.add_argument("--frame_inicio", type=int, help="fotograma en el que se elige al jugador (por defecto, su primera aparición)")
    ap.add_argument("--rol", default="delantero", help="rol del jugador para el informe")
    ap.add_argument("--desde", choices=FASES, default=None,
                    help="empezar en esta fase reutilizando lo guardado (por defecto: la primera que falte)")
    ap.add_argument("--hasta", choices=FASES, default="informe", help="parar después de esta fase")
    lado = ap.add_mutually_exclusive_group()
    lado.add_argument("--ataca_izquierda", action="store_true", help="forzar: su equipo ataca hacia la izquierda de la cámara")
    lado.add_argument("--ataca_derecha", action="store_true", help="forzar: su equipo ataca hacia la derecha de la cámara")
    ap.add_argument("--modelo_campo", default=None, help="pesos del modelo de puntos del campo (por defecto, el de Roboflow)")
    ap.add_argument("--campo", default=None,
                    help="medidas reales del campo LARGOxANCHO en metros, p. ej. 100x64 (por defecto 105x68)")
    ap.add_argument("--min_inliers", type=int, default=4, help="inliers mínimos para aceptar una homografía")
    ap.add_argument("--tau", type=float, default=TAU_POR_DEFECTO, help="umbral de re-identificación")
    ap.add_argument("--device", default=None, help="cuda o cpu (por defecto, cuda si hay GPU)")
    ap.add_argument("--silencio", action="store_true", help="menos mensajes de progreso")
    a = ap.parse_args(argv)

    if a.device is None:
        try:
            import torch
            a.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            a.device = "cpu"
    from src.fase2_homografia import comprobar_medidas, leer_medidas
    try:
        a.campo_medidas = leer_medidas(a.campo)
    except ValueError as e:
        print(f"✗ {e}"); return 1
    for aviso in comprobar_medidas(*a.campo_medidas):
        print(f"⚠ {aviso}")
    video = Path(a.video)
    if not video.exists():
        print(f"✗ No encuentro el vídeo {video}"); return 1
    salida = Path("outputs") / video.stem
    salida.mkdir(parents=True, exist_ok=True)

    # Por defecto se empieza en la primera fase cuyo resultado falta
    salidas = {"video": "homografia.csv", "huellas": "embeddings.npy", "equipos": "equipos.json",
               "jugador": None, "metricas": None, "informe": None}
    if a.desde is None:
        a.desde = next(f for f in FASES if salidas[f] is None or not (salida / salidas[f]).exists())
        if a.desde != "video":
            print(f"(Reutilizando lo guardado en {salida}: empiezo en la fase '{a.desde}'. "
                  f"Usa --desde video para repetirlo todo.)")
    fases = FASES[FASES.index(a.desde):FASES.index(a.hasta) + 1]

    ruta_resumen = salida / "resumen.json"
    resumen = json.load(open(ruta_resumen, encoding="utf-8")) if ruta_resumen.exists() else {}
    resumen.update(clip=video.name, device=a.device)
    textos = {"video": "Detectar jugadores y pasar el campo a metros (GPU, el paso más lento)",
              "huellas": "Huellas de apariencia de cada jugador (ResNet-18 + color)",
              "equipos": "Separar equipos y sentido de ataque",
              "jugador": "Seguir al jugador elegido (re-identificación)",
              "metricas": "Métricas y figuras", "informe": "Informe con el LLM"}
    t0 = time.time()
    try:
        for f in fases:
            paso(FASES.index(f) + 1, textos[f])
            t = time.time()
            if f == "metricas":
                resumen.update(fase_metricas(a, salida, resumen))
            else:
                resumen.update(globals()[f"fase_{f}"](a, salida))
            print(f"  ({time.time() - t:.0f} s)")
    except ErrorPipeline as e:
        print(f"\n✗ {e}")
        return 1
    finally:
        resumen["segundos_ultima_ejecucion"] = round(time.time() - t0, 1)
        json.dump(resumen, open(ruta_resumen, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print(f"\n✓ Hecho en {time.time() - t0:.0f} s. Resultados en {salida}/ (resumen en resumen.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
