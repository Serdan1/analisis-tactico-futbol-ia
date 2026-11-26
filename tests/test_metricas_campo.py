"""Las métricas de Alex usan las medidas del campo de campo.json (main.py --campo)."""
import json

import numpy as np
import pandas as pd

from src import fase2_metricas as M


def _posiciones(carpeta, x, y, n=100):
    df = pd.DataFrame({"frame": range(n), "t_s": np.arange(n) / 25, "track_id": 7, "equipo": "blanco",
                       "x_m": x, "y_m": y, "valido": 1})
    ruta = carpeta / "posiciones.csv"
    df.to_csv(ruta, index=False)
    return ruta


def test_sin_campo_json_usa_105x68(tmp_path):
    ruta = _posiciones(tmp_path, 95.0, 34.0)
    m = M.metricas_completas(ruta, {"id": 7})
    assert m["campo_m"] == [105.0, 68.0]
    assert m["zonas_pct"]["area_rival"] == 100.0 and m["zonas_pct"]["ultimo_tercio"] == 100.0


def test_campo_de_100x64(tmp_path):
    # x = 90 está en el área rival de un campo de 100 m (desde 83,5), y fuera del de 105 m (desde 88,5) no:
    # usamos x = 86, que solo está dentro del área si el campo mide 100 m.
    ruta = _posiciones(tmp_path, 86.0, 32.0)
    assert M.metricas_completas(ruta, {"id": 7})["zonas_pct"]["area_rival"] == 0.0
    (tmp_path / "campo.json").write_text(json.dumps({"largo_m": 100, "ancho_m": 64}))
    m = M.metricas_completas(ruta, {"id": 7})
    assert m["campo_m"] == [100.0, 64.0]
    assert m["zonas_pct"]["area_rival"] == 100.0
    assert m["zonas_pct"]["carril_central"] == 100.0       # y = 32 es el centro de un campo de 64 m


def test_figuras_con_otro_campo(tmp_path):
    from src import visualizaciones as V
    x = np.linspace(70, 85, 100) + np.random.default_rng(0).normal(0, 0.05, 100)
    ruta = _posiciones(tmp_path, x, 30.0)
    (tmp_path / "campo.json").write_text(json.dumps({"largo_m": 100, "ancho_m": 64}))
    figs = V.figuras_completas(ruta, tmp_path / "figuras")
    assert len(figs) == 6
