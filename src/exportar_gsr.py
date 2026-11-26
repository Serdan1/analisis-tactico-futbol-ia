"""Exportar nuestros resultados al formato oficial de SoccerNet Game State Reconstruction.

Sirve para calcular la métrica oficial GS-HOTA con el evaluador de SoccerNet
(sn-trackeval) y compararnos con su sistema base y con el reto de 2024.

GS-HOTA evalúa a TODAS las personas del campo a la vez (como un tracker
multi-objeto): para cada persona real, que haya una predicción a menos de 5 m
en el campo, con el mismo número a lo largo del tiempo y con los mismos
atributos: rol (player / goalkeeper / referee), equipo (left / right: el lado
que defiende) y dorsal. Nosotros no leemos dorsales: se exporta jersey = null.

Una predicción por detección con posición válida:
    track_id     -> nuestro número de ByteTrack
    bbox_pitch   -> metros con el origen en el centro (x hacia la derecha de la
                    cámara, y hacia la cámara): x = x_m − 52,5 ; y = 34 − y_m
    role         -> del detector (y portero / árbitro por track, como en equipos.csv)
    team         -> left si su equipo ataca hacia la derecha (defiende la izquierda)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LARGO, ANCHO = 105.0, 68.0
CATEGORIA = {"player": 1, "goalkeeper": 2, "referee": 3}


def exportar_clip(carpeta, ruta_gt, ruta_salida) -> int:
    """outputs/<clip>/ (posiciones_todos, equipos) -> <clip>.json de predicciones. Devuelve cuántas."""
    carpeta = Path(carpeta)
    gt = json.load(open(ruta_gt, encoding="utf-8"))
    imagen_de = {int(Path(im["file_name"]).stem) - 1: im["image_id"] for im in gt["images"]}

    todos = pd.read_csv(carpeta / "posiciones_todos.csv")
    eq = pd.read_csv(carpeta / "equipos.csv")[["frame", "track_id", "equipo", "es_portero"]]
    sentido = json.load(open(carpeta / "equipos.json", encoding="utf-8")).get("sentido", {})
    lado = {e: ("left" if s.get("ataca") == "derecha" else "right") for e, s in sentido.items()}

    d = todos.merge(eq, on=["frame", "track_id"], how="left")
    d = d[(d.valido == 1) & d.frame.isin(imagen_de)]
    preds = []
    for r in d.itertuples():
        if r.equipo == "arbitro" or r.clase == "arbitro":
            rol, equipo = "referee", None
        else:
            rol = "goalkeeper" if pd.notna(r.es_portero) and bool(r.es_portero) else "player"
            equipo = lado.get(r.equipo)
        x, y = float(r.x_m) - LARGO / 2, ANCHO / 2 - float(r.y_m)
        punto = {"x_bottom_left": x, "y_bottom_left": y, "x_bottom_middle": x, "y_bottom_middle": y,
                 "x_bottom_right": x, "y_bottom_right": y}
        preds.append({
            "image_id": imagen_de[int(r.frame)], "track_id": int(r.track_id), "supercategory": "object",
            "category_id": CATEGORIA[rol], "confidence": 1.0,
            "attributes": {"role": rol, "team": equipo, "jersey": None},
            "bbox_image": {"x": float(r.x1), "y": float(r.y1), "w": float(r.x2 - r.x1), "h": float(r.y2 - r.y1)},
            "bbox_pitch": punto,
        })
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"predictions": preds}, open(ruta_salida, "w", encoding="utf-8"))
    return len(preds)
