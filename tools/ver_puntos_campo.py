"""Ver qué puntos del campo detecta el modelo y cómo encaja nuestra homografía.

    python tools/ver_puntos_campo.py --video data/clips/SNGS-119.mp4 --frames 100 400 650
    python tools/ver_puntos_campo.py --video data/clips/SNGS-119.mp4 --frames 400 \
        --gt data/soccernet/SN-GSR-2025/SNGS-119/Labels-GameState.json

Guarda en outputs/figuras/puntos_<clip>_<frame>.jpg una imagen con:
  - los 32 puntos del modelo: verde = inlier de RANSAC, rojo = usado pero
    descartado, gris = confianza < 0,5 (no usado). Con su número (1-32).
  - en amarillo, las líneas del campo dibujadas con nuestra H: si la H es
    buena, caen encima de las líneas pintadas del césped.
  - con --gt: un aspa magenta en la posición REAL de cada jugador según
    SoccerNet, llevada a la imagen con nuestra H. Si la H es buena, cae en
    los pies del jugador.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fase2_homografia import ANCHO_M, CONFIG_CAMPO, LARGO_M, VERTICES_CM, EstimadorHomografia  # noqa: E402
from src.posiciones import MODELO_CAMPO, puntos_campo  # noqa: E402


def cm_a_px(puntos_cm, H):
    Hinv = np.linalg.inv(H)
    return cv2.perspectiveTransform(np.asarray(puntos_cm, np.float32).reshape(-1, 1, 2), Hinv).reshape(-1, 2)


def dibujar(frame, xy, conf, res, verdad=None):
    img = frame.copy()
    h, w = img.shape[:2]
    if res.H is not None:
        for a, b in CONFIG_CAMPO.edges:                      # líneas del campo con nuestra H
            t = np.linspace(0, 1, 30)[:, None]
            seg = VERTICES_CM[a - 1] * (1 - t) + VERTICES_CM[b - 1] * t
            px = cm_a_px(seg, res.H)
            ok = np.all(np.isfinite(px), axis=1) & (np.abs(px) < 1e5).all(axis=1)
            if ok.sum() > 1:
                cv2.polylines(img, [px[ok].astype(np.int32)], False, (0, 230, 255), 2)
        if verdad is not None and len(verdad):
            # SoccerNet: metros con origen en el centro. Los cm de la plantilla de
            # Roboflow tienen y = 0 en la banda lejana: y_cm = (34 + y_SoccerNet)·100
            cm = np.c_[(verdad.xg + LARGO_M / 2) * 100, (verdad.yg + ANCHO_M / 2) * 100]
            for (u, v) in cm_a_px(cm, res.H):
                if 0 <= u < w and 0 <= v < h:
                    cv2.drawMarker(img, (int(u), int(v)), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 3)
    for k, ((u, v), c) in enumerate(zip(xy, conf)):
        if c < 0.05 or u <= 1:
            continue
        color = (0, 200, 0) if res.mascara_inliers[k] else (0, 0, 255) if res.mascara_usados[k] else (150, 150, 150)
        cv2.circle(img, (int(u), int(v)), 8 if res.mascara_usados[k] else 5, color, -1)
        cv2.putText(img, f"{k + 1}", (int(u) + 9, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    txt = (f"puntos usados {res.n_puntos} · inliers {res.n_inliers} · error {res.error_cm:.0f} cm · "
           f"{'H valida' if res.valida else 'sin H'}{' (reutilizada)' if res.reutilizada else ''}")
    cv2.rectangle(img, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.putText(img, txt, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--gt", help="Labels-GameState.json de SoccerNet (opcional)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from ultralytics import YOLO
    modelo = YOLO(MODELO_CAMPO)
    verdad = None
    if a.gt:
        from src.evaluacion_soccernet import cargar_verdad
        verdad = cargar_verdad(a.gt)
    cap = cv2.VideoCapture(a.video)
    salida = Path("outputs/figuras"); salida.mkdir(parents=True, exist_ok=True)
    for f in a.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            print(f"✗ no se pudo leer el fotograma {f}"); continue
        xy, conf = puntos_campo(modelo, frame, a.device)
        res = EstimadorHomografia().actualizar(xy, conf)
        v = None if verdad is None else verdad[(verdad.frame == f) & verdad.xg.notna()]
        img = dibujar(frame, xy, conf, res, v)
        ruta = salida / f"puntos_{Path(a.video).stem}_{f}.jpg"
        cv2.imwrite(str(ruta), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{ruta}  ·  usados {res.n_puntos}, inliers {res.n_inliers}, error {res.error_cm:.0f} cm  ·  "
              f"puntos: {[k + 1 for k in np.flatnonzero(res.mascara_usados)]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
