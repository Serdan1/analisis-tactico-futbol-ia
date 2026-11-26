"""Exportación al formato oficial de SoccerNet GSR."""
import json

import pandas as pd

from src.exportar_gsr import exportar_clip


def test_exportar_clip(tmp_path):
    gt = {"images": [{"image_id": "3116000001", "file_name": "000001.jpg"},
                     {"image_id": "3116000002", "file_name": "000002.jpg"}], "annotations": []}
    json.dump(gt, open(tmp_path / "gt.json", "w"))
    pd.DataFrame([(0, 7, "jugador", 10, 20, 30, 80, 60.0, 30.0, 1),
                  (1, 7, "jugador", 11, 20, 31, 80, 61.0, 30.0, 1),
                  (1, 9, "arbitro", 50, 20, 70, 80, 52.5, 34.0, 1),
                  (1, 8, "jugador", 90, 20, 99, 80, None, None, 0)],      # sin posición: no se exporta
                 columns=["frame", "track_id", "clase", "x1", "y1", "x2", "y2", "x_m", "y_m", "valido"]
                 ).to_csv(tmp_path / "posiciones_todos.csv", index=False)
    pd.DataFrame([(0, 7, "blanco", False), (1, 7, "blanco", False), (1, 9, "arbitro", False), (1, 8, "azul", False)],
                 columns=["frame", "track_id", "equipo", "es_portero"]).to_csv(tmp_path / "equipos.csv", index=False)
    json.dump({"sentido": {"blanco": {"ataca": "izquierda"}, "azul": {"ataca": "derecha"}}},
              open(tmp_path / "equipos.json", "w"))
    n = exportar_clip(tmp_path, tmp_path / "gt.json", tmp_path / "out" / "SNGS-116.json")
    p = json.load(open(tmp_path / "out" / "SNGS-116.json"))["predictions"]
    assert n == 3
    j = p[0]
    assert j["image_id"] == "3116000001" and j["track_id"] == 7
    assert j["attributes"] == {"role": "player", "team": "right", "jersey": None}   # ataca izq. = defiende dcha.
    assert abs(j["bbox_pitch"]["x_bottom_middle"] - 7.5) < 1e-6                  # 60 − 52,5
    assert abs(j["bbox_pitch"]["y_bottom_middle"] - 4.0) < 1e-6                  # 34 − 30
    assert p[2]["attributes"]["role"] == "referee" and p[2]["attributes"]["team"] is None
