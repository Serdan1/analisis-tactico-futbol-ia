"""Pruebas de la re-identificación con datos sintéticos: python -m pytest tests/"""
import numpy as np
import pandas as pd

from src.fase1_tracking import calibrar_umbral, seguir_jugador


def cruce_sintetico():
    """Dos jugadores (blanco y oscuro) se cruzan; en el fotograma 10 el tracker les cambia el número
    y entre el 8 y el 9 el blanco está tapado (solo se ve el oscuro)."""
    rng = np.random.default_rng(0)
    base_blanco, base_oscuro = rng.normal(size=512), rng.normal(size=512)
    filas, embs = [], []
    for f in range(20):
        x = 50 + 0.2 * f
        for quien, y in (("blanco", 30.0), ("oscuro", 30.5)):
            if quien == "blanco" and f in (8, 9):
                continue
            tid = (7 if quien == "blanco" else 21) if f < 10 else (21 if quien == "blanco" else 7)
            e = (base_blanco if quien == "blanco" else base_oscuro) + rng.normal(0, 0.3, 512)
            filas.append(dict(frame=f, t_s=f / 25, track_id=tid, x_m=x, y_m=y, valido=1, quien=quien))
            embs.append(e / np.linalg.norm(e))
    return pd.DataFrame(filas), np.array(embs, np.float32)


def test_sigue_al_blanco_aunque_cambie_de_numero():
    todos, embs = cruce_sintetico()
    s = seguir_jugador(todos, embs, track_id=7, tau=0.8, n_galeria=3, sep_galeria=2)
    sigue = s.merge(todos, on=["frame", "track_id"], how="left")
    validos = sigue[sigue.valido_x == 1]
    assert (validos.quien == "blanco").all()                 # nunca coge al oscuro
    assert s.loc[s.frame.isin([8, 9]), "valido"].eq(0).all()  # tapado -> desconocido
    assert s.loc[s.frame >= 10, "track_id"].eq(21).all()      # tras el cruce sigue con su nuevo número
    assert s.cambio_id.sum() == 1


def test_calibrar_umbral_separa_bien():
    tau, tabla = calibrar_umbral(np.array([0.2, 0.3, 0.35]), np.array([0.9, 1.0, 1.1]))
    assert 0.35 <= tau < 0.9 and tabla.media.min() == 0
