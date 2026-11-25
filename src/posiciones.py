"""Vídeo -> posiciones en metros de todos los jugadores (tarea #6).

Une la detección y el seguimiento (Fase 1) con la homografía (Fase 2a).

Uso desde la carpeta del repositorio (rutas RELATIVAS, por las tildes de la ruta):

    python -m src.posiciones --video data/clips/2e57b9_0.mp4 --salida outputs/2e57b9_0
    python -m src.posiciones --video data/clips/2e57b9_0.mp4 --salida outputs/2e57b9_0 --track_id 11

Genera en la carpeta de salida:
    posiciones_todos.csv   una fila por jugador y fotograma
    posiciones.csv         solo el jugador --track_id, con el formato de docs/formatos.md
    homografia.csv         diagnóstico por fotograma (puntos usados, inliers, error)
    control.mp4            (con --video_control) vídeo con los puntos del campo y un radar
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from trackers import ByteTrackTracker
from ultralytics import YOLO

from src.fase2_homografia import (ANCHO_M, LARGO_M, VERTICES_CM, EstimadorHomografia,
                                  dentro_del_campo, pies_de_cajas, pixel_a_metros)

MODELO_JUGADORES = "models/football-player-detection.pt"
MODELO_CAMPO = "models/football-pitch-detection.pt"
CLASES = {0: "balon", 1: "portero", 2: "jugador", 3: "arbitro"}


def puntos_campo(modelo_campo, frame, device):
    """Devuelve los 32 puntos del campo (px) y su confianza."""
    r = modelo_campo(frame, device=device, verbose=False)[0]
    if r.keypoints is None or r.keypoints.conf is None or len(r.keypoints.xy) == 0:
        return np.zeros((32, 2), np.float32), np.zeros(32, np.float32)
    return r.keypoints.xy[0].cpu().numpy(), r.keypoints.conf[0].cpu().numpy()


def procesar_video(video, device="cuda", imgsz=1280, confianza_min=0.5, usar_ransac=True,
                   max_frames=None, paso=1, video_control=None, progreso=True, modo_demo=False):
    """Recorre el vídeo y devuelve dos DataFrames: posiciones de todos y diagnóstico de H.

    modo_demo=True reproduce la demo de Roboflow (todos los puntos, sin RANSAC
    y sin descartar homografías malas), para comparar en la memoria.
    """
    error_max_cm = 150.0
    if modo_demo:
        confianza_min, usar_ransac, error_max_cm = 0.0, False, float("inf")
    modelo_jug = YOLO(MODELO_JUGADORES)
    modelo_campo = YOLO(MODELO_CAMPO)
    info = sv.VideoInfo.from_video_path(str(video))
    fps = info.fps
    tracker = ByteTrackTracker(minimum_consecutive_frames=3, track_activation_threshold=0.25,
                               frame_rate=fps / paso)
    estimador = EstimadorHomografia(confianza_min=confianza_min, usar_ransac=usar_ransac,
                                    error_max_cm=error_max_cm,
                                    max_frames_reutilizar=max(1, round(fps / paso)))
    filas, diag = [], []

    escritor = None
    if video_control:
        escritor = cv2.VideoWriter(str(video_control), cv2.VideoWriter_fourcc(*"mp4v"),
                                   fps / paso, (info.width, info.height))

    frames = sv.get_video_frames_generator(str(video), stride=paso)
    total = info.total_frames // paso if max_frames is None else min(max_frames, info.total_frames // paso)
    for i, frame in enumerate(frames):
        if max_frames is not None and i >= max_frames:
            break
        n_frame = i * paso
        t_s = n_frame / fps

        # 1. Homografía del fotograma
        kp_xy, kp_conf = puntos_campo(modelo_campo, frame, device)
        res = estimador.actualizar(kp_xy, kp_conf)
        diag.append(dict(frame=n_frame, t_s=round(t_s, 3), valida=int(res.valida),
                         reutilizada=int(res.reutilizada), n_puntos=res.n_puntos,
                         n_inliers=res.n_inliers, error_cm=round(res.error_cm, 1)))

        # 2. Detección y seguimiento (sin el balón)
        r = modelo_jug(frame, imgsz=imgsz, device=device, verbose=False)[0]
        det = sv.Detections.from_ultralytics(r)
        det = det[det.class_id != 0]
        det = tracker.update(det)
        det = det[det.tracker_id != -1]

        # 3. Pies de cada jugador -> metros
        xy_m = np.full((len(det), 2), np.nan, np.float32)
        if res.valida and len(det):
            xy_m = pixel_a_metros(pies_de_cajas(det.xyxy), res.H)
        validos = np.zeros(len(det), bool) if not res.valida else dentro_del_campo(xy_m)
        for k in range(len(det)):
            filas.append(dict(frame=n_frame, t_s=round(t_s, 3), track_id=int(det.tracker_id[k]),
                              clase=CLASES.get(int(det.class_id[k]), "?"),
                              x_m=round(float(xy_m[k, 0]), 2) if validos[k] else np.nan,
                              y_m=round(float(xy_m[k, 1]), 2) if validos[k] else np.nan,
                              valido=int(validos[k])))

        if escritor is not None:
            escritor.write(dibujar_control(frame, kp_xy, res, det, xy_m, validos))
        if progreso and (i % 25 == 0):
            print(f"  fotograma {i}/{total}  puntos={res.n_puntos} inliers={res.n_inliers} "
                  f"error={res.error_cm:.0f} cm  jugadores={len(det)}", flush=True)

    if escritor is not None:
        escritor.release()
    return pd.DataFrame(filas), pd.DataFrame(diag)


def posiciones_de_un_jugador(todos: pd.DataFrame, diag: pd.DataFrame, track_id: int,
                             equipo: str = "", ataca_izquierda: bool = False) -> pd.DataFrame:
    """Formato de docs/formatos.md: una fila por fotograma; vacío si no se ve.

    ataca_izquierda=True gira el campo (x -> 105 - x, y -> 68 - y) para que el
    equipo del jugador ataque siempre hacia x = 105, como pide el formato.
    """
    j = todos[todos.track_id == track_id].set_index("frame")
    out = diag[["frame", "t_s"]].copy()
    out["track_id"] = track_id
    out["equipo"] = equipo
    out["x_m"] = out.frame.map(j.x_m)
    out["y_m"] = out.frame.map(j.y_m)
    out["valido"] = out.frame.map(j.valido).fillna(0).astype(int)
    if ataca_izquierda:
        out["x_m"] = (LARGO_M - out.x_m).round(2)
        out["y_m"] = (ANCHO_M - out.y_m).round(2)
    return out


def posiciones_por_tramos(todos: pd.DataFrame, diag: pd.DataFrame, tramos, equipo: str = "",
                          ataca_izquierda: bool = False) -> pd.DataFrame:
    """Une varios track_id que en realidad son el mismo jugador (arreglo manual de un ID switch).

    tramos: lista de (track_id, frame_inicio, frame_fin), ambos incluidos; None = sin límite.
    Ejemplo en 2e57b9_0.mp4: el delantero blanco es el 7 hasta el fotograma 281 y el 21
    desde el 288 (entre medias está tapado por el defensa y el tracker les cambia el número):
        posiciones_por_tramos(todos, diag, [(7, None, 281), (21, 288, None)], ataca_izquierda=True)
    """
    partes = []
    for tid, ini, fin in tramos:
        t = todos[todos.track_id == tid]
        if ini is not None:
            t = t[t.frame >= ini]
        if fin is not None:
            t = t[t.frame <= fin]
        partes.append(t)
    union = pd.concat(partes).sort_values("frame")
    if union.frame.duplicated().any():
        raise ValueError("Los tramos se solapan: revisa los fotogramas de inicio y fin")
    union = union.assign(track_id=tramos[0][0])
    pos = posiciones_de_un_jugador(union, diag, tramos[0][0], equipo, ataca_izquierda)
    pos["track_id"] = "+".join(str(t[0]) for t in tramos)
    return pos


# ---------------------------------------------------------------- vídeo de control
def dibujar_control(frame, kp_xy, res, det, xy_m, validos, escala=6, margen=30):
    """Fotograma con los puntos del campo (verde = inlier, amarillo = usado pero
    descartado por RANSAC, rojo = descartado por baja confianza) y un radar."""
    img = frame.copy()
    for idx, (u, v) in enumerate(kp_xy):
        if u <= 1 and v <= 1:
            continue
        color = (0, 200, 0) if res.mascara_inliers[idx] else \
                (0, 215, 255) if res.mascara_usados[idx] else (0, 0, 220)
        cv2.circle(img, (int(u), int(v)), 9, color, -1)
        cv2.putText(img, str(idx + 1), (int(u) + 10, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for k, (x1, y1, x2, y2) in enumerate(det.xyxy.astype(int)):
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(img, str(det.tracker_id[k]), (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    radar = dibujar_radar(xy_m[validos], det.tracker_id[validos], escala, margen)
    h, w = radar.shape[:2]
    H_img, W_img = img.shape[:2]
    y0, x0 = H_img - h - 10, (W_img - w) // 2
    img[y0:y0 + h, x0:x0 + w] = cv2.addWeighted(img[y0:y0 + h, x0:x0 + w], 0.25, radar, 0.75, 0)
    estado = "H nueva" if res.valida and not res.reutilizada else "H reutilizada" if res.valida else "SIN H"
    cv2.putText(img, f"{estado} | puntos {res.n_puntos} | inliers {res.n_inliers} | error {res.error_cm:.0f} cm",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    return img


def dibujar_radar(xy_m, ids, escala=6, margen=30):
    """Campo 2D visto desde arriba (y = 0 abajo, como en docs/formatos.md)."""
    W, H = int(LARGO_M * escala) + 2 * margen, int(ANCHO_M * escala) + 2 * margen
    radar = np.full((H, W, 3), (60, 140, 60), np.uint8)

    def p(x, y):  # metros -> píxel del radar
        return int(margen + x * escala), int(H - margen - y * escala)

    blanco = (255, 255, 255)
    cv2.rectangle(radar, p(0, ANCHO_M), p(LARGO_M, 0), blanco, 2)
    cv2.line(radar, p(LARGO_M / 2, 0), p(LARGO_M / 2, ANCHO_M), blanco, 2)
    cv2.circle(radar, p(LARGO_M / 2, ANCHO_M / 2), int(9.15 * escala), blanco, 2)
    for x0, s in ((0, 1), (LARGO_M, -1)):
        cv2.rectangle(radar, p(x0, ANCHO_M / 2 + 20.16), p(x0 + s * 16.5, ANCHO_M / 2 - 20.16), blanco, 2)
        cv2.rectangle(radar, p(x0, ANCHO_M / 2 + 9.16), p(x0 + s * 5.5, ANCHO_M / 2 - 9.16), blanco, 2)
    for (x, y), i in zip(xy_m, ids):
        cv2.circle(radar, p(x, y), 7, (0, 140, 255), -1)
        cv2.putText(radar, str(i), (p(x, y)[0] + 8, p(x, y)[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, blanco, 1)
    return radar


def main():
    ap = argparse.ArgumentParser(description="Vídeo -> posiciones en metros")
    ap.add_argument("--video", required=True, help="ruta RELATIVA al clip")
    ap.add_argument("--salida", required=True, help="carpeta de salida, p. ej. outputs/clip_01")
    ap.add_argument("--track_id", type=int, help="jugador para posiciones.csv (de momento a mano; el clic llega en #9)")
    ap.add_argument("--ataca_izquierda", action="store_true",
                    help="el equipo del jugador ataca hacia la portería de la izquierda de la cámara")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--confianza_min", type=float, default=0.5)
    ap.add_argument("--sin_ransac", action="store_true")
    ap.add_argument("--modo_demo", action="store_true",
                    help="como la demo de Roboflow: todos los puntos y sin RANSAC (para comparar)")
    ap.add_argument("--max_frames", type=int)
    ap.add_argument("--paso", type=int, default=1, help="procesar 1 de cada N fotogramas")
    ap.add_argument("--video_control", action="store_true", help="guardar control.mp4")
    a = ap.parse_args()

    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    todos, diag = procesar_video(a.video, a.device, a.imgsz, a.confianza_min, not a.sin_ransac,
                                 a.max_frames, a.paso, salida / "control.mp4" if a.video_control else None,
                                 modo_demo=a.modo_demo)
    todos.to_csv(salida / "posiciones_todos.csv", index=False)
    diag.to_csv(salida / "homografia.csv", index=False)
    print(f"\nH válida en {diag.valida.mean():.0%} de los fotogramas "
          f"({diag.reutilizada.sum()} reutilizando la anterior). Error medio {diag.error_cm.median():.0f} cm (mediana).")
    print(f"Jugadores distintos (track_id): {todos.track_id.nunique()}")
    if a.track_id is not None:
        pos = posiciones_de_un_jugador(todos, diag, a.track_id, ataca_izquierda=a.ataca_izquierda)
        pos.to_csv(salida / "posiciones.csv", index=False)
        print(f"posiciones.csv del jugador {a.track_id}: {pos.valido.mean():.0%} de fotogramas válidos")
    print(f"Resultados en {salida}/")


if __name__ == "__main__":
    main()
