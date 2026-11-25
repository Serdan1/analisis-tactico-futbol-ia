"""Pruebas de la homografía con datos sintéticos: python -m pytest tests/"""
import cv2
import numpy as np

from src.fase2_homografia import (ANCHO_M, VERTICES_CM, EstimadorHomografia,
                                  pixel_a_metros)


def camara_sintetica():
    """Homografía campo (cm) -> imagen (px) parecida a un plano de TV."""
    campo = np.float32([[0, 0], [10500, 0], [10500, 6800], [0, 6800]])
    imagen = np.float32([[300, 350], [1650, 350], [2300, 1100], [-380, 1100]])
    return cv2.getPerspectiveTransform(campo, imagen)


def proyectar(puntos, M):
    return cv2.perspectiveTransform(np.float32(puntos).reshape(-1, 1, 2), M).reshape(-1, 2)


def test_recupera_posiciones_y_descarta_puntos_inventados():
    rng = np.random.default_rng(0)
    G = camara_sintetica()
    kp = proyectar(VERTICES_CM, G) + rng.normal(0, 2, (32, 2))      # ruido de 2 px
    conf = np.ones(32, np.float32)
    falsos = np.arange(13)                                           # como en la demo
    kp[falsos] = rng.uniform([0, 300], [500, 700], (13, 2))
    conf[falsos] = 0.0

    jugador_cm = np.float32([[6000, 2000]])                          # x = 60 m, a 20 m de la banda lejana
    jugador_px = proyectar(jugador_cm, G)

    # Con filtro de confianza + RANSAC
    res = EstimadorHomografia(confianza_min=0.5).actualizar(kp, conf)
    assert res.valida and res.n_puntos == 19
    xy = pixel_a_metros(jugador_px, res.H)[0]
    assert abs(xy[0] - 60) < 0.5 and abs(xy[1] - (ANCHO_M - 20)) < 0.5

    # Como la demo (sin filtro ni RANSAC): el error es mucho mayor
    demo = EstimadorHomografia(confianza_min=0.0, usar_ransac=False, error_max_cm=1e9).actualizar(kp, conf)
    xy_demo = pixel_a_metros(jugador_px, demo.H)[0]
    assert np.linalg.norm(xy_demo - xy) > 3


def test_reutiliza_la_ultima_h_y_luego_la_invalida():
    G = camara_sintetica()
    kp, conf = proyectar(VERTICES_CM, G), np.ones(32, np.float32)
    est = EstimadorHomografia(max_frames_reutilizar=2)
    assert est.actualizar(kp, conf).valida
    sin = np.zeros(32, np.float32)
    assert est.actualizar(kp, sin).reutilizada
    assert est.actualizar(kp, sin).reutilizada
    assert not est.actualizar(kp, sin).valida


def test_girar_campo_si_ataca_a_la_izquierda():
    import pandas as pd
    from src.posiciones import posiciones_de_un_jugador
    diag = pd.DataFrame({"frame": [0, 1], "t_s": [0.0, 0.04]})
    todos = pd.DataFrame({"frame": [0, 1], "track_id": [7, 7], "x_m": [10.0, 20.0],
                          "y_m": [5.0, 60.0], "valido": [1, 1]})
    p = posiciones_de_un_jugador(todos, diag, 7, ataca_izquierda=True)
    assert p.x_m.tolist() == [95.0, 85.0] and p.y_m.tolist() == [63.0, 8.0]


def test_unir_tramos_de_un_id_switch():
    import pandas as pd
    from src.posiciones import posiciones_por_tramos
    diag = pd.DataFrame({"frame": range(6), "t_s": [i * 0.04 for i in range(6)]})
    todos = pd.DataFrame({"frame": [0, 1, 2, 0, 1, 2, 4, 5], "track_id": [7, 7, 7, 21, 21, 21, 21, 21],
                          "x_m": [1, 2, 3, 50, 50, 50, 4, 5], "y_m": [0] * 8, "valido": [1] * 8})
    p = posiciones_por_tramos(todos, diag, [(7, None, 2), (21, 4, None)])
    assert p.x_m.tolist()[:3] == [1, 2, 3] and p.x_m.tolist()[4:] == [4, 5]
    assert p.valido.tolist() == [1, 1, 1, 0, 1, 1] and p.track_id.iloc[0] == "7+21"
