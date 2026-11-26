"""Evaluación con SoccerNet: un clip sintético en formato Labels-GameState.json."""
import json

import numpy as np
import pandas as pd

from src.evaluacion_soccernet import evaluar_clip


def _clip(tmp_path, n=60):
    imgs, anns, filas = [], [], []
    for f in range(n):
        iid = f"9{f:05d}"
        imgs.append({"image_id": iid, "file_name": f"{f + 1:06d}.jpg"})
        for gid, lado, x0 in [(1, "left", 200), (2, "right", 600)]:
            x = x0 + 3 * f
            anns.append({"image_id": iid, "track_id": gid, "category_id": 1,
                         "attributes": {"role": "player", "team": lado, "jersey": str(gid)},
                         "bbox_image": {"x": x, "y": 400, "w": 30, "h": 70},
                         "bbox_pitch": {"x_bottom_middle": -20 + gid * 1.0 + 0.1 * f, "y_bottom_middle": 5.0}})
            # el tracker intercambia los números en el 30, con los dos jugadores a 1 m
            # (los intercambios de ByteTrack ocurren cuando se tapan, no a distancia)
            tid = gid if f < 30 else 3 - gid
            filas.append((f, f / 25, tid, "jugador", x, 400, x + 30, 470,
                          52.5 - 20 + gid * 1.0 + 0.1 * f + 0.2, 34 - 5.0, 1, gid))
    json.dump({"images": imgs, "annotations": anns}, open(tmp_path / "gt.json", "w"))
    todos = pd.DataFrame(filas, columns=["frame", "t_s", "track_id", "clase", "x1", "y1", "x2", "y2",
                                         "x_m", "y_m", "valido", "real"])
    E = np.zeros((len(todos), 640), np.float32)
    E[np.arange(len(todos)), todos.real.to_numpy()] = 1           # la apariencia sí sabe quién es quién
    out = tmp_path / "out"; out.mkdir()
    todos.drop(columns="real").to_csv(out / "posiciones_todos.csv", index=False)
    np.save(out / "embeddings.npy", E)
    return out, tmp_path / "gt.json"


def test_evaluar_clip_sintetico(tmp_path):
    out, gt = _clip(tmp_path)
    r = evaluar_clip(out, gt, tau=0.5, min_frames=20)
    assert r["transformacion"] == "x, -y"
    assert r["detectadas_pct"] == 100.0
    assert abs(r["error_mediana_m"] - 0.2) < 0.01
    assert r["tracking"]["cambios_de_numero"] == 2                 # uno por jugador
    s = r["seguimiento"]
    assert s["reid"]["bien_pct"] > 95                               # la re-ID corrige el cambio
    assert s["solo_tracker"]["otro_pct"] > 40                       # seguir el número no
