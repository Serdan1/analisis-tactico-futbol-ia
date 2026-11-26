"""Evaluar nuestro sistema con la verdad de SoccerNet Game State Reconstruction (tarea #25).

SoccerNet GSR trae, en cada fotograma de cada clip, la caja de cada persona en
la imagen y su posición real en el campo (metros, origen en el centro), con su
equipo (left / right: el lado que defiende), rol y dorsal.

Lo que medimos (sin tocar ningún parámetro de nuestro sistema):

1. Detección: % de personas reales que detectamos (emparejando cajas por IoU ≥ 0,5).
2. Posición: error en metros entre nuestra x_m, y_m y la real, en las parejas
   emparejadas por la caja. Así se mide solo la homografía + los pies, no la
   detección.
3. Tracking: cambios de número de ByteTrack (cuántas veces el número nuestro
   de un jugador real cambia).
4. Re-identificación: para cada jugador real, lo elegimos en su primer
   fotograma y lo seguimos con seguir_jugador(). En cada fotograma, ¿la
   posición que damos es la suya, la de otro o ninguna? Se compara con
   seguir solo el número del tracker.
5. Equipos y sentido de ataque frente a left / right.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

LARGO, ANCHO = 105.0, 68.0
ROLES = {1: "jugador", 2: "portero", 3: "arbitro"}


# --------------------------------------------------------------- leer la verdad
def cargar_verdad(ruta_json) -> pd.DataFrame:
    """Labels-GameState.json -> una fila por persona y fotograma (sin balón).

    frame empieza en 0 (como en nuestro vídeo); xg, yg en metros con el origen
    en el centro del campo, tal como vienen (se pasan a nuestro sistema con
    transformar()).
    """
    j = json.load(open(ruta_json, encoding="utf-8"))
    frame_de = {im["image_id"]: int(Path(im["file_name"]).stem) - 1 for im in j["images"]}
    filas = []
    for a in j["annotations"]:
        if a.get("category_id") not in ROLES or a.get("bbox_image") is None:
            continue
        b, p, at = a["bbox_image"], a.get("bbox_pitch") or {}, a.get("attributes") or {}
        filas.append((frame_de[a["image_id"]], int(a["track_id"]), ROLES[a["category_id"]],
                      at.get("team"), at.get("jersey"),
                      b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"],
                      p.get("x_bottom_middle", np.nan), p.get("y_bottom_middle", np.nan)))
    return pd.DataFrame(filas, columns=["frame", "gt_id", "rol", "lado", "dorsal",
                                        "x1", "y1", "x2", "y2", "xg", "yg"])


TRANSFORMACIONES = {          # (signo de x, signo de y): nuestro = (sx·x + 52,5 ; 34 − sy·y)
    "x, -y": (1, 1), "x, y": (1, -1), "-x, -y": (-1, 1), "-x, y": (-1, -1)}


def transformar(xg, yg, sx, sy):
    return sx * np.asarray(xg) + LARGO / 2, ANCHO / 2 - sy * np.asarray(yg)


# --------------------------------------------------------------- emparejar por cajas
def iou_matriz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda c: (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def emparejar(todos: pd.DataFrame, verdad: pd.DataFrame, iou_min: float = 0.5) -> pd.DataFrame:
    """Parejas (índice nuestro, índice verdad) por fotograma con el algoritmo húngaro sobre 1 − IoU."""
    parejas = []
    nuestros = dict(tuple(todos.groupby("frame")))
    for f, g in verdad.groupby("frame"):
        n = nuestros.get(f)
        if n is None or n.empty:
            continue
        iou = iou_matriz(n[["x1", "y1", "x2", "y2"]].to_numpy(float), g[["x1", "y1", "x2", "y2"]].to_numpy(float))
        fi, co = linear_sum_assignment(-iou)
        ok = iou[fi, co] >= iou_min
        parejas += list(zip(n.index[fi[ok]], g.index[co[ok]], iou[fi[ok], co[ok]]))
    return pd.DataFrame(parejas, columns=["i_nuestro", "i_verdad", "iou"])


# --------------------------------------------------------------- métricas
def evaluar_posiciones(todos, verdad, parejas):
    m = parejas.join(todos[["frame", "track_id", "x_m", "y_m", "valido"]], on="i_nuestro") \
               .join(verdad[["gt_id", "rol", "lado", "xg", "yg"]], on="i_verdad")
    m = m[m.xg.notna()]
    v = m[m.valido.astype(bool)]
    errores = {}
    for nombre, (sx, sy) in TRANSFORMACIONES.items():
        xr, yr = transformar(v.xg, v.yg, sx, sy)
        errores[nombre] = float(np.median(np.hypot(v.x_m - xr, v.y_m - yr))) if len(v) else np.inf
    mejor = min(errores, key=errores.get)
    sx, sy = TRANSFORMACIONES[mejor]
    m["xr"], m["yr"] = transformar(m.xg, m.yg, sx, sy)
    m["error_m"] = np.hypot(m.x_m - m.xr, m.y_m - m.yr)
    e = m.loc[m.valido.astype(bool), "error_m"]
    res = dict(
        transformacion=mejor,
        personas_reales=int(len(verdad)),
        detectadas_pct=round(100 * parejas.i_verdad.nunique() / max(len(verdad), 1), 1),
        detecciones_nuestras_emparejadas_pct=round(100 * parejas.i_nuestro.nunique() / max(len(todos), 1), 1),
        con_posicion_pct=round(100 * m.valido.astype(bool).mean(), 1) if len(m) else 0.0,
        error_mediana_m=round(float(e.median()), 2) if len(e) else None,
        error_p90_m=round(float(e.quantile(.9)), 2) if len(e) else None,
        error_p95_m=round(float(e.quantile(.95)), 2) if len(e) else None,
        error_menos_1m_pct=round(100 * float((e < 1).mean()), 1) if len(e) else None,
        error_menos_2m_pct=round(100 * float((e < 2).mean()), 1) if len(e) else None,
    )
    return res, m


def cambios_de_numero(m: pd.DataFrame) -> dict:
    """Para cada jugador real, cuántas veces cambia el número que le da ByteTrack."""
    cambios, tracks = 0, 0
    for _, g in m.sort_values("frame").groupby("gt_id"):
        ids = g.track_id.to_numpy()
        cambios += int((ids[1:] != ids[:-1]).sum())
        tracks += 1
    return dict(jugadores_reales=tracks, cambios_de_numero=cambios,
                cambios_por_jugador=round(cambios / max(tracks, 1), 2))


def evaluar_seguimiento(todos, embs, m, tau=0.80, min_frames=250, max_jugadores=None):
    """Elegir a cada jugador real en su primer fotograma y seguirlo durante el clip.

    Para cada fotograma en que el jugador real aparece: "bien" si la posición
    que damos es de una detección emparejada con él; "otro" si es de otra
    persona (o de una detección sin pareja); "sin dato" si no damos posición.
    Se compara la re-identificación con seguir solo el número inicial del tracker.
    """
    from .fase1_tracking import seguir_jugador
    de_quien = dict(zip(zip(m.frame, m.track_id), m.gt_id))        # (frame, nuestro id) -> gt_id
    jug = m[m.rol == "jugador"]
    filas = []
    candidatos = jug.groupby("gt_id").frame.agg(["min", "count"])
    candidatos = candidatos[candidatos["count"] >= min_frames].sort_values("count", ascending=False)
    if max_jugadores:
        candidatos = candidatos.head(max_jugadores)
    for gid, (f0, _) in candidatos.iterrows():
        tid = int(jug[(jug.gt_id == gid) & (jug.frame == f0)].track_id.iloc[0])
        sus = set(m[m.gt_id == gid].frame)
        seg = seguir_jugador(todos, embs, tid, tau, frame_inicio=int(f0))
        solo = todos[todos.track_id == tid].set_index("frame").valido.astype(bool)
        for metodo, elegido in [("reid", dict(zip(seg.frame[seg.valido == 1], seg.track_id[seg.valido == 1]))),
                                ("solo_tracker", {f: tid for f in solo.index[solo]})]:
            bien = otro = nada = 0
            for f in sus:
                if f not in elegido:
                    nada += 1
                elif de_quien.get((f, elegido[f])) == gid:
                    bien += 1
                else:
                    otro += 1
            # fotogramas en que damos posición de otro aunque él no esté detectado
            otro += sum(1 for f in elegido if f not in sus and f >= f0
                        and de_quien.get((f, elegido[f])) not in (None, gid))
            filas.append(dict(gt_id=int(gid), metodo=metodo, frames=len(sus), bien=bien, otro=otro, sin_dato=nada))
    df = pd.DataFrame(filas)
    if df.empty:
        return {}, df
    res = {}
    for metodo, g in df.groupby("metodo"):
        tot = (g.bien + g.otro + g.sin_dato).sum()
        res[metodo] = dict(jugadores=int(len(g)), bien_pct=round(100 * g.bien.sum() / tot, 1),
                           otro_pct=round(100 * g.otro.sum() / tot, 1),
                           sin_dato_pct=round(100 * g.sin_dato.sum() / tot, 1),
                           jugadores_sin_error=int((g.otro == 0).sum()))
    return res, df


def evaluar_equipos(m: pd.DataFrame, equipos: pd.DataFrame, sentido: dict, sx: int) -> dict:
    e = m.merge(equipos[["frame", "track_id", "equipo"]], on=["frame", "track_id"])
    e = e[(e.rol == "jugador") & e.lado.isin(["left", "right"]) & ~e.equipo.isin(["", "arbitro", "otro"])]
    if e.empty:
        return {}
    tabla = pd.crosstab(e.equipo, e.lado)
    a_lado = tabla.idxmax(axis=1).to_dict()                         # color -> left/right
    acierto = float((e.equipo.map(a_lado) == e.lado).mean())
    # el equipo "left" defiende la portería de x = -52,5 en SoccerNet
    ataca_real = {"left": "derecha" if sx == 1 else "izquierda", "right": "izquierda" if sx == 1 else "derecha"}
    sentido_ok = {c: (sentido.get(c, {}).get("ataca") == ataca_real[l]) for c, l in a_lado.items() if c in sentido}
    return dict(colores=a_lado, colores_distintos=len(set(a_lado.values())) == len(a_lado),
                acierto_pct=round(100 * acierto, 1),
                sentido_correcto=all(sentido_ok.values()) if sentido_ok else None,
                sentido_metodo={c: s.get("metodo") for c, s in sentido.items()})


def evaluar_clip(carpeta_salida, ruta_json, tau=0.80, con_reid=True, max_jugadores=None,
                 min_frames=250) -> dict:
    carpeta_salida = Path(carpeta_salida)
    todos = pd.read_csv(carpeta_salida / "posiciones_todos.csv")
    verdad = cargar_verdad(ruta_json)
    parejas = emparejar(todos, verdad)
    res, m = evaluar_posiciones(todos, verdad, parejas)
    res["tracking"] = cambios_de_numero(m)
    sx = TRANSFORMACIONES[res["transformacion"]][0]
    if (carpeta_salida / "equipos.csv").exists():
        sentido = json.load(open(carpeta_salida / "equipos.json", encoding="utf-8"))["sentido"]
        res["equipos"] = evaluar_equipos(m, pd.read_csv(carpeta_salida / "equipos.csv"), sentido, sx)
    if con_reid and (carpeta_salida / "embeddings.npy").exists():
        embs = np.load(carpeta_salida / "embeddings.npy")
        res["seguimiento"], detalle = evaluar_seguimiento(todos, embs, m, tau, min_frames=min_frames,
                                                             max_jugadores=max_jugadores)
        detalle.to_csv(carpeta_salida / "evaluacion_seguimiento.csv", index=False)
    m.to_csv(carpeta_salida / "evaluacion_parejas.csv", index=False)
    json.dump(res, open(carpeta_salida / "evaluacion.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return res
