"""Fase 2a: de píxeles a metros sobre el campo.

Responsable: Serdan1
Salida: posiciones en metros con el formato de docs/formatos.md

Cómo funciona
-------------
1. Un modelo YOLO-pose (football-pitch-detection.pt, de roboflow/sports)
   localiza en cada fotograma los 32 puntos característicos del campo.
2. Nos quedamos SOLO con los puntos con confianza alta. El modelo devuelve
   siempre los 32, también los que no se ven, y esos vienen "inventados"
   (lo vimos en la demo, docs/demo_roboflow/).
3. Con al menos 4 puntos calculamos la homografía H (3x3) con RANSAC, que
   descarta además los puntos mal detectados.
4. Si en un fotograma no hay suficientes puntos fiables, reutilizamos la
   última H válida durante un máximo de fotogramas.
5. Con H proyectamos los pies del jugador (centro inferior de su caja) al
   plano del campo.

Sistema de coordenadas (docs/formatos.md)
-----------------------------------------
Campo de LARGO x ANCHO m (por defecto 105 x 68, la medida recomendada por la
FIFA). x va de 0 a LARGO a lo largo del campo, de izquierda a derecha tal como
lo ve la cámara. y va de 0 (banda del lado de la cámara) a ANCHO (banda lejana).

Medidas del campo
-----------------
Las Reglas de Juego permiten campos de distintas medidas, pero las áreas, el
punto de penalti y el círculo central miden siempre lo mismo. Por eso
config_campo() cambia solo el largo y el ancho y deja las áreas fijas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from sports.configs.soccer import SoccerPitchConfiguration

LARGO_M = 105.0   # medidas por defecto (recomendación de la FIFA y de la mayoría de estadios de 1ª)
ANCHO_M = 68.0

# Límites de las Reglas de Juego (IFAB, regla 1)
LIMITES_REGLAS = {"largo": (90.0, 120.0), "ancho": (45.0, 90.0)}
LIMITES_INTERNACIONAL = {"largo": (100.0, 110.0), "ancho": (64.0, 75.0)}


def comprobar_medidas(largo_m: float, ancho_m: float) -> list[str]:
    """Lanza ValueError si las medidas no son posibles según las Reglas de Juego.

    Devuelve avisos (lista de textos) si son válidas pero no de partido internacional.
    """
    (l0, l1), (a0, a1) = LIMITES_REGLAS["largo"], LIMITES_REGLAS["ancho"]
    if not (l0 <= largo_m <= l1 and a0 <= ancho_m <= a1):
        raise ValueError(f"Un campo de {largo_m:g} x {ancho_m:g} m no cumple las Reglas de Juego "
                         f"(largo {l0:g}-{l1:g} m, ancho {a0:g}-{a1:g} m)")
    if ancho_m >= largo_m:
        raise ValueError("El largo tiene que ser mayor que el ancho (¿están al revés?)")
    (l0, l1), (a0, a1) = LIMITES_INTERNACIONAL["largo"], LIMITES_INTERNACIONAL["ancho"]
    if not (l0 <= largo_m <= l1 and a0 <= ancho_m <= a1):
        return [f"{largo_m:g} x {ancho_m:g} m es válido, pero fuera de las medidas de partido "
                f"internacional ({l0:g}-{l1:g} x {a0:g}-{a1:g} m)"]
    return []


def leer_medidas(texto: str | None) -> tuple[float, float]:
    """'100x64' -> (100.0, 64.0). None o '' -> medidas por defecto."""
    if not texto:
        return LARGO_M, ANCHO_M
    try:
        largo, ancho = (float(v.replace(",", ".")) for v in texto.lower().replace("×", "x").split("x"))
    except ValueError:
        raise ValueError(f"No entiendo las medidas '{texto}': escribe LARGOxANCHO en metros, p. ej. 100x64")
    comprobar_medidas(largo, ancho)
    return largo, ancho


def config_campo(largo_m: float = LARGO_M, ancho_m: float = ANCHO_M) -> SoccerPitchConfiguration:
    """Plantilla de los 32 puntos en centímetros. Áreas, penalti y círculo reglamentarios y fijos.

    La configuración por defecto de roboflow/sports mide 120 x 70 m y su área no es reglamentaria.
    """
    return SoccerPitchConfiguration(
        length=int(round(largo_m * 100)), width=int(round(ancho_m * 100)),
        penalty_box_length=1650, penalty_box_width=4032,
        goal_box_length=550, goal_box_width=1832,
        centre_circle_radius=915, penalty_spot_distance=1100,
    )


CONFIG_CAMPO = config_campo()
VERTICES_CM = np.array(CONFIG_CAMPO.vertices, dtype=np.float32)  # (32, 2), campo por defecto


@dataclass
class ResultadoHomografia:
    """Lo que sabemos de la homografía de un fotograma."""
    H: np.ndarray | None          # matriz 3x3 imagen (px) -> campo (cm), o None
    valida: bool                  # True si se puede usar para proyectar
    reutilizada: bool             # True si es la de un fotograma anterior
    n_puntos: int                 # puntos que pasaron el filtro de confianza
    n_inliers: int                # puntos que RANSAC dio por buenos
    error_cm: float               # error medio de reproyección de los inliers (cm)
    mascara_usados: np.ndarray = field(default_factory=lambda: np.zeros(32, bool))
    mascara_inliers: np.ndarray = field(default_factory=lambda: np.zeros(32, bool))


class EstimadorHomografia:
    """Calcula la homografía de cada fotograma y la mantiene entre fotogramas.

    Parámetros
    ----------
    confianza_min : solo se usan puntos con confianza >= este valor.
        Con 0.0 se reproduce el comportamiento de la demo de Roboflow
        (útil para comparar en la memoria).
    usar_ransac : RANSAC descarta puntos mal detectados. Sin él, mínimos
        cuadrados con todos los puntos (como la demo).
    umbral_ransac_cm : distancia máxima (en cm sobre el campo) para que un
        punto cuente como inlier.
    max_frames_reutilizar : cuántos fotogramas seguidos se puede reutilizar
        la última H válida antes de marcar las posiciones como no válidas.
    error_max_cm : si el error medio de reproyección supera este valor, la H
        del fotograma se descarta.
    """

    def __init__(self, confianza_min: float = 0.5, usar_ransac: bool = True,
                 umbral_ransac_cm: float = 100.0, max_frames_reutilizar: int = 25,
                 error_max_cm: float = 150.0, min_inliers: int = 4,
                 largo_m: float = LARGO_M, ancho_m: float = ANCHO_M):
        self.confianza_min = confianza_min
        comprobar_medidas(largo_m, ancho_m)
        self.largo_m, self.ancho_m = largo_m, ancho_m
        self.vertices_cm = np.array(config_campo(largo_m, ancho_m).vertices, dtype=np.float32)
        # Con 4 puntos la homografía encaja siempre (8 ecuaciones, 8 incógnitas):
        # el error sale 0 aunque los puntos estén mal. Exigir más inliers da
        # redundancia para detectarlo (experimento de la #17).
        self.min_inliers = min_inliers
        self.usar_ransac = usar_ransac
        self.umbral_ransac_cm = umbral_ransac_cm
        self.max_frames_reutilizar = max_frames_reutilizar
        self.error_max_cm = error_max_cm
        self._ultima_H: np.ndarray | None = None
        self._frames_sin_H = 0

    def reiniciar(self):
        self._ultima_H = None
        self._frames_sin_H = 0

    def actualizar(self, puntos_px: np.ndarray, confianzas: np.ndarray) -> ResultadoHomografia:
        """Calcula la H de un fotograma a partir de sus 32 puntos del campo.

        puntos_px : (32, 2) posiciones en píxeles que devuelve el modelo.
        confianzas : (32,) confianza de cada punto.
        """
        puntos_px = np.asarray(puntos_px, dtype=np.float32)
        confianzas = np.asarray(confianzas, dtype=np.float32)
        usados = (confianzas >= self.confianza_min) & (puntos_px[:, 0] > 1) & (puntos_px[:, 1] > 1)
        n = int(usados.sum())

        H, inliers, error = None, np.zeros(32, bool), float("nan")
        if n >= 4:
            src, dst = puntos_px[usados], self.vertices_cm[usados]
            metodo = cv2.RANSAC if self.usar_ransac else 0
            H, mask = cv2.findHomography(src, dst, metodo, self.umbral_ransac_cm)
            if H is not None:
                # Sin RANSAC, OpenCV marca como inliers solo los puntos por debajo
                # del umbral; pero la H se ha calculado con todos, así que todos cuentan.
                if self.usar_ransac and mask is not None:
                    m = mask.ravel().astype(bool)
                else:
                    m = np.ones(n, bool)
                inliers[np.flatnonzero(usados)[m]] = True
                proy = cv2.perspectiveTransform(src[m].reshape(-1, 1, 2), H).reshape(-1, 2)
                error = float(np.linalg.norm(proy - dst[m], axis=1).mean())
                if not np.isfinite(error) or error > self.error_max_cm or int(m.sum()) < self.min_inliers:
                    H = None

        if H is not None:
            self._ultima_H, self._frames_sin_H = H, 0
            return ResultadoHomografia(H, True, False, n, int(inliers.sum()), error, usados, inliers)

        # Sin H nueva: reutilizar la última durante un tiempo limitado
        self._frames_sin_H += 1
        if self._ultima_H is not None and self._frames_sin_H <= self.max_frames_reutilizar:
            return ResultadoHomografia(self._ultima_H, True, True, n, int(inliers.sum()), error, usados, inliers)
        return ResultadoHomografia(None, False, False, n, int(inliers.sum()), error, usados, inliers)


def pixel_a_metros(puntos_px: np.ndarray, H: np.ndarray, ancho_m: float = ANCHO_M) -> np.ndarray:
    """Convierte puntos (u, v) en píxeles a (x, y) en metros con nuestro sistema de ejes.

    puntos_px : (N, 2). Devuelve (N, 2) en metros. ancho_m: el mismo que usó la homografía.
    """
    puntos_px = np.asarray(puntos_px, dtype=np.float32).reshape(-1, 1, 2)
    if len(puntos_px) == 0:
        return np.empty((0, 2), dtype=np.float32)
    cm = cv2.perspectiveTransform(puntos_px, H).reshape(-1, 2)
    x_m = cm[:, 0] / 100.0
    y_m = ancho_m - cm[:, 1] / 100.0   # Roboflow pone y = 0 en la banda lejana
    return np.stack([x_m, y_m], axis=1)


def dentro_del_campo(xy_m: np.ndarray, margen_m: float = 3.0, largo_m: float = LARGO_M,
                     ancho_m: float = ANCHO_M) -> np.ndarray:
    """True para posiciones dentro del campo, con un margen para jugadores en la banda."""
    xy_m = np.asarray(xy_m)
    return ((xy_m[:, 0] >= -margen_m) & (xy_m[:, 0] <= largo_m + margen_m) &
            (xy_m[:, 1] >= -margen_m) & (xy_m[:, 1] <= ancho_m + margen_m))


def pies_de_cajas(xyxy: np.ndarray) -> np.ndarray:
    """Centro inferior de cada caja (x1, y1, x2, y2): el punto que toca el césped."""
    xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
    return np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], axis=1)
