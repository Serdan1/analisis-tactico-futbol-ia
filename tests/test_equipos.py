"""Pruebas de la separación por equipos (#10) con datos sintéticos."""
import numpy as np
import pandas as pd

from src.fase1_equipos import (ClasificadorEquipos, asignar_equipos, nombre_de_color,
                               proponer_etiquetas, sentido_de_ataque)


def _partido_sintetico(semilla=0, n_frames=120):
    """Equipo A (huella ~ +e1) en x < 52, equipo B (~ +e2) en x > 52.

    Track 0 (A) y track 1 (B) se intercambian el número en el fotograma 60,
    como el 7 y el 21 del clip real. Portero 90 junto a x = 3 (del B, que
    defiende la izquierda), árbitro 99.
    """
    rng = np.random.default_rng(semilla)
    base = {"A": np.eye(640)[0], "B": np.eye(640)[1], "P": np.eye(640)[2], "R": np.eye(640)[3]}
    filas, embs, verdad = [], [], []
    jugadores = {i: ("A" if i < 5 else "B") for i in range(10)}
    for f in range(n_frames):
        for tid, eq in jugadores.items():
            real = eq
            if tid in (0, 1) and f >= 60:           # cambio de número
                real = "B" if tid == 0 else "A"
            # B defiende la izquierda (su portero está en x=3): su bloque está más atrás
            x = 55 + rng.normal(0, 8) if real == "A" else 35 + rng.normal(0, 8)
            filas.append((f, tid, "jugador", x)); embs.append(base[real]); verdad.append(real)
        filas.append((f, 90, "portero", 3.0)); embs.append(base["P"]); verdad.append("P")
        filas.append((f, 99, "arbitro", 50.0)); embs.append(base["R"]); verdad.append("R")
    todos = pd.DataFrame(filas, columns=["frame", "track_id", "clase", "x_m"])
    todos["y_m"] = 34.0
    todos["valido"] = 1
    embs = np.array(embs) + rng.normal(0, 0.05, (len(embs), 640))
    return todos, embs.astype(np.float32), np.array(verdad)


def test_equipos_con_cambio_de_numero_y_sentido():
    todos, embs, verdad = _partido_sintetico()
    prop = proponer_etiquetas(todos, embs, n_por_equipo=10, frame_max=50, partes="ambos")
    assert len(prop) == 20
    # las etiquetas propuestas por K-Means coinciden con la verdad (dentro de cada grupo)
    for g in (0, 1):
        assert len(set(verdad[prop[prop.grupo == g].indice])) == 1
    y = np.where(verdad[prop.indice] == "A", "A", "B")
    clf = ClasificadorEquipos("knn", "ambos", k=3).fit(embs[prop.indice], y)
    out = asignar_equipos(todos, embs, clf, ventana=9)

    jug = (todos.clase == "jugador").to_numpy()
    assert (out.equipo.to_numpy()[jug] == verdad[jug]).mean() > 0.97
    # el track 0 es del A antes del cambio y del B después
    t0 = out[out.track_id == 0].set_index("frame").equipo
    assert t0.loc[:55].eq("A").all() and t0.loc[65:].eq("B").all()
    assert out[out.track_id == 99].equipo.eq("arbitro").all()
    assert out[out.track_id == 90].equipo.eq("B").all()          # portero junto a su equipo
    sentido = sentido_de_ataque(out)
    assert sentido["B"]["ataca"] == "derecha" and sentido["B"]["metodo"] == "portero"
    assert sentido["A"]["ataca"] == "izquierda"


def test_logreg_tambien_funciona():
    todos, embs, verdad = _partido_sintetico(1)
    prop = proponer_etiquetas(todos, embs, n_por_equipo=10, frame_max=50, partes="ambos")
    y = np.where(verdad[prop.indice] == "A", "A", "B")
    clf = ClasificadorEquipos("logreg", "ambos").fit(embs[prop.indice], y)
    jug = (todos.clase == "jugador").to_numpy()
    assert (clf.predict(embs[jug]) == verdad[jug]).mean() > 0.97


def test_nombres_de_color():
    # valores HSV (OpenCV) medidos en los dos clips de prueba
    assert nombre_de_color(34, 13, 245) == "blanco"    # 2e57b9_0, camiseta blanca
    assert nombre_de_color(94, 51, 114) == "azul"      # 2e57b9_0, azul marino
    assert nombre_de_color(99, 28, 223) == "blanco"    # 08fd33_0, blanco a la sombra
    assert nombre_de_color(53, 108, 242) == "verde"    # 08fd33_0, verde lima (antes salía "claro")
    assert nombre_de_color(0, 0, 30) == "negro"
    assert nombre_de_color(175, 200, 200) == "rojo"
