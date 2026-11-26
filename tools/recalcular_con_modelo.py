"""Repetir la evaluación completa de SoccerNet con otro modelo de puntos del campo (tarea #17).

    python tools/recalcular_con_modelo.py --modelo models/puntos_congelado.pt
    python tools/recalcular_con_modelo.py --modelo models/puntos_congelado.pt --min_inliers 6

Sin volver a pasar YOLO de jugadores ni ResNet: reutiliza de outputs/<clip>/ las
cajas, los números del tracker y las huellas (no dependen de la homografía) y
solo recalcula lo que sí depende de ella: posiciones en metros -> equipos y
sentido de ataque -> seguimiento / re-identificación -> evaluación.
Los puntos del campo salen de la caché de tools/evaluar_homografia.py
(outputs/<clip>/puntos_<modelo>.npz) o se calculan si no está.

Resultados en outputs/<modelo>_mi<N>/<clip>/ y la tabla comparada con el
modelo original en outputs/<modelo>_mi<N>/soccernet_resultados.csv.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluar_homografia import puntos_del_clip  # noqa: E402
from evaluar_soccernet import ORIGEN, fila_resumen  # noqa: E402
from src.evaluacion_soccernet import evaluar_clip  # noqa: E402
from src.fase1_equipos import equipos_automaticos  # noqa: E402
from src.fase2_homografia import EstimadorHomografia, dentro_del_campo, pixel_a_metros  # noqa: E402

COLUMNAS = ["error_mediana_m", "error_p95_m", "error_menos_1m_pct", "con_posicion_pct", "equipos_acierto_pct",
            "reid_bien_pct", "reid_otro_pct", "tracker_bien_pct", "tracker_otro_pct", "reid_precision_pct"]


def recalcular(base: Path, destino: Path, xy, conf, min_inliers):
    todos = pd.read_csv(base / "posiciones_todos.csv")
    est = EstimadorHomografia(min_inliers=min_inliers)
    diag, Hs = [], {}
    for f in range(len(xy)):
        res = est.actualizar(xy[f], conf[f])
        Hs[f] = res.H if res.valida else None
        diag.append(dict(frame=f, t_s=round(f / 25, 3), valida=int(res.valida), reutilizada=int(res.reutilizada),
                         n_puntos=res.n_puntos, n_inliers=res.n_inliers, error_cm=round(res.error_cm, 1)))
    todos[["x_m", "y_m"]] = np.nan
    todos["valido"] = 0
    for f, idx in todos.groupby("frame").indices.items():
        H = Hs.get(f)
        if H is None:
            continue
        g = todos.iloc[idx]
        pies = np.c_[(g.x1 + g.x2) / 2, g.y2].astype(np.float32)
        m = pixel_a_metros(pies, H)
        ok = dentro_del_campo(m)
        cols = todos.columns.get_indexer(["x_m", "y_m", "valido"])
        todos.iloc[idx[ok], cols[0]] = np.round(m[ok, 0], 2)
        todos.iloc[idx[ok], cols[1]] = np.round(m[ok, 1], 2)
        todos.iloc[idx, cols[2]] = ok.astype(int)
    destino.mkdir(parents=True, exist_ok=True)
    todos.to_csv(destino / "posiciones_todos.csv", index=False)
    pd.DataFrame(diag).to_csv(destino / "homografia.csv", index=False)
    shutil.copy(base / "embeddings.npy", destino / "embeddings.npy")
    return todos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--min_inliers", type=int, default=4)
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--margen_cambio", type=float, default=None, help="seguidor: margen para cambiar de número en un conflicto")
    ap.add_argument("--max_hueco_s", type=float, default=None, help="seguidor (modo prudente): no recuperar tras más de N s perdido")
    a = ap.parse_args()

    nombre = Path(a.modelo).stem
    seguir_kw = {k: v for k, v in [("margen_cambio", a.margen_cambio), ("max_hueco_s", a.max_hueco_s)] if v is not None}
    sufijo = "".join(f"_{k}{v}" for k, v in seguir_kw.items())
    salida = Path("outputs") / f"{nombre}_mi{a.min_inliers}{sufijo}"
    clips = a.clips or sorted(p.parent.name for p in Path("outputs").glob("SNGS-*/evaluacion_parejas.csv"))
    filas = []
    for k, clip in enumerate(clips, 1):
        base = Path("outputs") / clip
        video = Path("data/clips") / f"{clip}.mp4"
        xy, conf = puntos_del_clip(video, a.modelo, base / f"puntos_{nombre}.npz", a.device)
        destino = salida / clip
        todos = recalcular(base, destino, xy, conf, a.min_inliers)
        embs = np.load(destino / "embeddings.npy")
        try:
            eq, sentido, _ = equipos_automaticos(video, todos, embs)
            eq[["frame", "track_id", "clase", "equipo", "es_portero", "prob"]].to_csv(destino / "equipos.csv", index=False)
            json.dump({"sentido": sentido}, open(destino / "equipos.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  ⚠ equipos: {e}")
        r = evaluar_clip(destino, ORIGEN / clip / "Labels-GameState.json", seguir_kw=seguir_kw)
        f = fila_resumen(clip, r)
        b, o = f.get("reid_bien_pct") or 0, f.get("reid_otro_pct") or 0
        f["reid_precision_pct"] = round(100 * b / (b + o), 1) if b + o else None   # de las posiciones dadas, % del jugador correcto
        filas.append(f)
        print(f"[{k}/{len(clips)}] {clip}: error {f['error_mediana_m']} m · equipos {f['equipos_acierto_pct']} % · "
              f"sentido {'ok' if f['sentido_ok'] else 'MAL'} · seguimiento re-ID {f['reid_bien_pct']} % "
              f"(otro {f['reid_otro_pct']} %) / tracker {f['tracker_bien_pct']} %", flush=True)

    nuevo = pd.DataFrame(filas)
    nuevo.to_csv(salida / "soccernet_resultados.csv", index=False)
    comp = {"modelo nuevo": nuevo[COLUMNAS].mean()}
    if Path("outputs/soccernet_resultados.csv").exists():
        viejo = pd.read_csv("outputs/soccernet_resultados.csv")
        viejo["reid_precision_pct"] = 100 * viejo.reid_bien_pct / (viejo.reid_bien_pct + viejo.reid_otro_pct)
        viejo = viejo[viejo["clip"].isin(nuevo["clip"])]
        comp = {"Roboflow": viejo[COLUMNAS].mean(), **comp}
    tabla = pd.DataFrame(comp).round(1)
    tabla["sentido_ok (clips)"] = ""
    print("\nMedia de los clips:\n" + tabla.drop(columns="sentido_ok (clips)").to_string())
    print(f"Sentido de ataque correcto en {int(nuevo.sentido_ok.fillna(False).sum())} de {len(nuevo)} clips")
    print(f"\nGuardado en {salida / 'soccernet_resultados.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
