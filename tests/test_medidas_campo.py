"""Campos de distintas medidas: las áreas son fijas, solo cambian largo y ancho."""
import cv2
import numpy as np
import pytest

from src.fase2_homografia import (EstimadorHomografia, config_campo, leer_medidas,
                                  comprobar_medidas, pixel_a_metros)


def test_areas_fijas_y_bandas_variables():
    v105 = np.array(config_campo(105, 68).vertices)
    v100 = np.array(config_campo(100, 64).vertices)
    # Área izquierda (puntos 10-13, índices 9-12): misma distancia a la línea de fondo
    assert np.allclose(v105[9:13, 0], 1650) and np.allclose(v100[9:13, 0], 1650)
    # Y mismo tamaño del área a lo ancho (40,32 m) aunque el campo sea más estrecho
    assert np.isclose(v105[12, 1] - v105[9, 1], 4032) and np.isclose(v100[12, 1] - v100[9, 1], 4032)
    # La línea de medio campo (punto 14) sí se mueve
    assert v105[13, 0] == 5250 and v100[13, 0] == 5000


def test_campo_de_100x64_con_la_plantilla_correcta_y_con_la_de_105x68():
    """Un campo real de 100 x 64: con --campo 100x64 sale exacto; con 105 x 68 hay metros de error."""
    real = np.float32(config_campo(100, 64).vertices)
    G = cv2.getPerspectiveTransform(np.float32([[0, 0], [10000, 0], [10000, 6400], [0, 6400]]),
                                    np.float32([[300, 350], [1650, 350], [2300, 1100], [-380, 1100]]))
    kp = cv2.perspectiveTransform(real.reshape(-1, 1, 2), G).reshape(-1, 2)
    conf = np.ones(32, np.float32)
    jugador_cm = np.float32([[9000, 1000]])           # a 10 m de la línea de fondo, a 10 m de la banda lejana
    jugador_px = cv2.perspectiveTransform(jugador_cm.reshape(-1, 1, 2), G).reshape(-1, 2)

    bien = EstimadorHomografia(largo_m=100, ancho_m=64).actualizar(kp, conf)
    xy = pixel_a_metros(jugador_px, bien.H, ancho_m=64)[0]
    assert abs(xy[0] - 90) < 0.05 and abs(xy[1] - 54) < 0.05

    # Con la plantilla de 105 x 68, los puntos no encajan entre sí: o se descarta la H o el error es de metros
    mal = EstimadorHomografia(umbral_ransac_cm=1e4, error_max_cm=1e9).actualizar(kp, conf)
    xy_mal = pixel_a_metros(jugador_px, mal.H)[0]
    assert np.hypot(xy_mal[0] - 90, xy_mal[1] - 54) > 1.0


def test_leer_y_comprobar_medidas():
    assert leer_medidas(None) == (105.0, 68.0)
    assert leer_medidas("100x64") == (100.0, 64.0)
    assert leer_medidas("102,5×66") == (102.5, 66.0)
    assert comprobar_medidas(105, 68) == []
    assert comprobar_medidas(95, 60)            # válido, pero no internacional: avisa
    with pytest.raises(ValueError):
        leer_medidas("130x70")                  # fuera de las Reglas de Juego
    with pytest.raises(ValueError):
        leer_medidas("68x105")                  # al revés
    with pytest.raises(ValueError):
        leer_medidas("cien")
