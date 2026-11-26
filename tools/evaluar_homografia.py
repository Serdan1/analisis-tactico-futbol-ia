"""Comparar modelos de puntos del campo por su error en metros en los clips de test (tarea #17).

    python tools/evaluar_homografia.py                                    # solo el modelo de Roboflow
    python tools/evaluar_homografia.py --modelos models/puntos_congelado.pt models/puntos_completo.pt

Necesita haber pasado antes tools/evaluar_soccernet.py por esos clips: reutiliza
nuestras cajas de jugadores (posiciones_todos.csv) y su emparejamiento con la
verdad (evaluacion_parejas.csv). Así solo cambia la homografía y la
comparación es justa. Por cada modelo pasa la red de puntos por cada
fotograma (se guarda en outputs/<clip>/puntos_<modelo>.npz para no repetirlo) y
prueba varios mínimos de inliers.

Salida: outputs/homografia_comparacion.csv (por modelo) y _por_clip.csv.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fase2_homografia import EstimadorHomografia, pixel_a_metros  # noqa: E402
from src.posiciones import MODELO_CAMPO  # noqa: E402


def puntos_del_clip(video, ruta_modelo, cache, device):
    if cache.exists():
        d = np.load(cache)
        return d["xy"], d["conf"]
    from ultralytics import YOLO
    modelo = YOLO(ruta_modelo)
    cap = cv2.VideoCapture(str(video))
    xs, cs = [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        r = modelo(f, device=device, verbose=False)[0]
        if r.keypoints is None or r.keypoints.conf is None or len(r.keypoints.xy) == 0:
            xs.append(np.zeros((32, 2), np.float32)); cs.append(np.zeros(32, np.float32))
        else:
            xs.append(r.keypoints.xy[0].cpu().numpy()); cs.append(r.keypoints.conf[0].cpu().numpy())
    xy, conf = np.stack(xs), np.stack(cs)
    np.savez_compressed(cache, xy=xy, conf=conf)
    return xy, conf


def errores(xy, conf, todos, parejas, min_inliers):
    est = EstimadorHomografia(min_inliers=min_inliers)
    Hs = [est.actualizar(xy[f], conf[f]).H for f in range(len(xy))]
    p = parejas.join(todos[["x1", "x2", "y2"]], on="i_nuestro")
    out = np.full(len(p), np.nan)
    for f, idx in p.groupby("frame").indices.items():
        if f >= len(Hs) or Hs[f] is None:
            continue
        g = p.iloc[idx]
        pies = np.c_[(g.x1 + g.x2) / 2, g.y2]
        m = pixel_a_metros(pies, Hs[f])
        out[idx] = np.hypot(m[:, 0] - g.xr, m[:, 1] - g.yr)
    return out


def resumen(e):
    v = e[np.isfinite(e)]
    if not len(v):
        return dict(con_H_pct=0.0)
    return dict(con_H_pct=round(100 * len(v) / len(e), 1), mediana_m=round(float(np.median(v)), 2),
                p90_m=round(float(np.quantile(v, .9)), 2), menos_1m_pct=round(100 * float((v < 1).mean()), 1),
                menos_2m_pct=round(100 * float((v < 2).mean()), 1),
                menos_5m_pct=round(100 * float((v < 5).mean()), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="*", default=[], help="pesos además del de Roboflow")
    ap.add_argument("--min_inliers", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    clips = a.clips or sorted(p.parent.name for p in Path("outputs").glob("SNGS-*/evaluacion_parejas.csv"))
    if not clips:
        print("✗ Primero: python tools/evaluar_soccernet.py --n 10"); return 1
    modelos = [("roboflow", MODELO_CAMPO)] + [(Path(m).stem, m) for m in a.modelos]

    filas, todas = [], {}
    for k, clip in enumerate(clips, 1):
        salida = Path("outputs") / clip
        todos = pd.read_csv(salida / "posiciones_todos.csv")
        parejas = pd.read_csv(salida / "evaluacion_parejas.csv")[["frame", "i_nuestro", "xr", "yr"]].reset_index(drop=True)
        for nombre, ruta in modelos:
            xy, conf = puntos_del_clip(Path("data/clips") / f"{clip}.mp4", ruta, salida / f"puntos_{nombre}.npz", a.device)
            for mi in a.min_inliers:
                e = errores(xy, conf, todos, parejas, mi)
                todas.setdefault((nombre, mi), []).append(e)
                filas.append(dict(clip=clip, modelo=nombre, min_inliers=mi, **resumen(e)))
        hecho = [f for f in filas if f["clip"] == clip and f["min_inliers"] == a.min_inliers[0]]
        print(f"[{k}/{len(clips)}] {clip}: " + " · ".join(f"{f['modelo']} {f.get('mediana_m')} m" for f in hecho), flush=True)

    por_clip = pd.DataFrame(filas)
    total = pd.DataFrame([dict(modelo=n, min_inliers=mi, **resumen(np.concatenate(v))) for (n, mi), v in todas.items()])
    por_clip.to_csv("outputs/homografia_comparacion_por_clip.csv", index=False)
    total.to_csv("outputs/homografia_comparacion.csv", index=False)
    print("\nError de posición de los jugadores (todas las parejas de todos los clips):")
    print(total.to_string(index=False))
    print("\nGuardado en outputs/homografia_comparacion.csv y outputs/homografia_comparacion_por_clip.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
