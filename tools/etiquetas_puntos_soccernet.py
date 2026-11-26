"""Etiquetas de los 32 puntos del campo a partir de SoccerNet GSR (tarea #17).

    python tools/etiquetas_puntos_soccernet.py            # split train, 1 de cada 10 fotogramas

Idea: SoccerNet da, para cada jugador, dónde están sus pies en la imagen (px) y
en el campo (m). Esos puntos del suelo definen la homografía real del
fotograma (con 20-60 parejas; error de reproyección ~0 px). Con ella se
calcula dónde caen en la imagen los 32 puntos del campo de la plantilla de
Roboflow: son las etiquetas para reentrenar su modelo YOLOv8-pose, sin marcar
nada a mano.

Salida en data/puntos_campo/ (formato YOLO-pose):
    images/{train,val}/SNGS-XXX_000123.jpg
    labels/{train,val}/SNGS-XXX_000123.txt    "0 cx cy w h  x1 y1 v1 ... x32 y32 v32" (normalizado)
    puntos_campo.yaml
    muestras.jpg   (unas cuantas imágenes con sus etiquetas dibujadas, para revisar)
La validación son clips distintos de los de entrenamiento (los últimos --val_clips).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fase2_homografia import ANCHO_M, LARGO_M, VERTICES_CM  # noqa: E402

ORIGEN = Path("data/soccernet/SN-GSR-2025")
DESTINO = Path("data/puntos_campo")


def parejas_por_fotograma(ruta_json):
    """{frame: (px (N,2), cm (N,2))} con los pies (izq., centro, dcha.) de cada persona.

    Se usa bbox_pitch_raw (proyección directa de la cámara anotada, sin el
    suavizado temporal de bbox_pitch) para que la homografía sea exacta.
    SoccerNet: metros, origen en el centro, y hacia la cámara. Plantilla de
    Roboflow: cm, origen en la esquina lejana -> x_cm = (x + 52,5)·100, y_cm = (34 + y)·100.
    """
    j = json.load(open(ruta_json, encoding="utf-8"))
    frame_de = {im["image_id"]: int(Path(im["file_name"]).stem) - 1 for im in j["images"]}
    por = {}
    for a in j["annotations"]:
        b, p = a.get("bbox_image"), a.get("bbox_pitch_raw")
        if a.get("category_id") not in (1, 2, 3) or not b or not p:
            continue
        suelo = b["y"] + b["h"]
        for u, lado in [(b["x"], "left"), (b["x"] + b["w"] / 2, "middle"), (b["x"] + b["w"], "right")]:
            x, y = p.get(f"x_bottom_{lado}"), p.get(f"y_bottom_{lado}")
            if x is None or y is None:
                continue
            por.setdefault(frame_de[a["image_id"]], []).append((u, suelo, (x + LARGO_M / 2) * 100, (y + ANCHO_M / 2) * 100))
    return {f: (np.float32(v)[:, :2], np.float32(v)[:, 2:]) for f, v in por.items()}


def proyectar(H, pts_cm):
    """cm -> px con la homografía; devuelve también w (w <= 0: detrás de la cámara)."""
    h = np.c_[pts_cm, np.ones(len(pts_cm))] @ H.T
    return h[:, :2] / h[:, 2:3], h[:, 2]


def etiqueta(H, ancho, alto, margen=2):
    px, w = proyectar(H, VERTICES_CM)
    vis = (w > 0) & (px[:, 0] >= margen) & (px[:, 0] < ancho - margen) & (px[:, 1] >= margen) & (px[:, 1] < alto - margen)
    if vis.sum() < 4:
        return None, vis, px
    # caja del objeto "campo": la parte del rectángulo del campo que se ve en la imagen
    t = np.linspace(0, 1, 200)[:, None]
    L, A = LARGO_M * 100, ANCHO_M * 100
    borde = np.vstack([np.c_[t * L, t * 0], np.c_[t * L, t * 0 + A], np.c_[t * 0, t * A], np.c_[t * 0 + L, t * A]])
    bp, bw = proyectar(H, borde)
    bp = bp[bw > 0]
    x1, y1 = np.clip(bp.min(0), 0, [ancho, alto]); x2, y2 = np.clip(bp.max(0), 0, [ancho, alto])
    x1, y1 = min(x1, px[vis, 0].min()), min(y1, px[vis, 1].min())
    x2, y2 = max(x2, px[vis, 0].max()), max(y2, px[vis, 1].max())
    campos = [0, (x1 + x2) / 2 / ancho, (y1 + y2) / 2 / alto, (x2 - x1) / ancho, (y2 - y1) / alto]
    for (u, v), ok in zip(px, vis):
        campos += [u / ancho, v / alto, 2] if ok else [0, 0, 0]
    return " ".join(f"{c:.6f}" if isinstance(c, float) else str(c) for c in campos), vis, px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--cada", type=int, default=10, help="un fotograma de cada N")
    ap.add_argument("--val_clips", type=int, default=7, help="clips que se reservan para validación")
    ap.add_argument("--min_personas", type=int, default=6)
    a = ap.parse_args()

    info = json.load(open(ORIGEN / "sequences_info.json"))
    nombres = [s["name"] for s in info["train" if a.split == "train" else a.split]]
    clips = [c for c in nombres if (ORIGEN / c / "Labels-GameState.json").exists()]
    if not clips:
        print(f"✗ No hay clips del split {a.split} en {ORIGEN}: ejecuta tools/descargar_soccernet.py --split {a.split}")
        return 1
    val = set(clips[-a.val_clips:]) if a.val_clips > 0 else set()
    for p in ["images/train", "images/val", "labels/train", "labels/val"]:
        (DESTINO / p).mkdir(parents=True, exist_ok=True)

    cuenta = {"train": 0, "val": 0}
    saltados = 0
    muestras = []
    vis_total = np.zeros(32)
    for k, clip in enumerate(clips, 1):
        parejas = parejas_por_fotograma(ORIGEN / clip / "Labels-GameState.json")
        parte = "val" if clip in val else "train"
        primera = next((ORIGEN / clip / "img1").glob("*.jpg"))
        alto, ancho = cv2.imread(str(primera)).shape[:2]
        for f in [f for f in sorted(parejas) if f % a.cada == 0]:
            px, cm = parejas[f]
            if len(px) < 3 * a.min_personas:
                saltados += 1; continue
            H, _ = cv2.findHomography(cm, px, 0)
            if H is None:
                saltados += 1; continue
            res = np.linalg.norm(proyectar(H, cm)[0] - px, axis=1)
            if np.median(res) > 3:                     # anotación incoherente: mejor no usarla
                saltados += 1; continue
            img = ORIGEN / clip / "img1" / f"{f + 1:06d}.jpg"
            linea, vis, kp = etiqueta(H, ancho, alto)
            if linea is None:
                saltados += 1; continue
            nombre = f"{clip}_{f + 1:06d}"
            shutil.copy(img, DESTINO / "images" / parte / f"{nombre}.jpg")
            (DESTINO / "labels" / parte / f"{nombre}.txt").write_text(linea + "\n")
            cuenta[parte] += 1
            vis_total += vis
            if len(muestras) < 12 and f % 250 == 0:
                muestras.append((img, kp, vis))
        print(f"  [{k}/{len(clips)}] {clip} ({parte}) · imágenes: train {cuenta['train']}, val {cuenta['val']}", flush=True)

    (DESTINO / "puntos_campo.yaml").write_text(
        f"path: {DESTINO.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        "kpt_shape: [32, 3]\nnames:\n  0: pitch\n", encoding="utf-8")

    tiles = []
    for img, kp, vis in muestras:
        im = cv2.imread(str(img))
        for i, ((u, v), ok) in enumerate(zip(kp, vis)):
            if ok:
                cv2.circle(im, (int(u), int(v)), 10, (0, 255, 0), -1)
                cv2.putText(im, str(i + 1), (int(u) + 12, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        tiles.append(cv2.resize(im, (640, 360)))
    if tiles:
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(DESTINO / "muestras.jpg"), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))

    print(f"\n✓ {cuenta['train']} imágenes de entrenamiento y {cuenta['val']} de validación "
          f"({len(val)} clips aparte) · {saltados} fotogramas descartados")
    print(f"  Puntos visibles por imagen (media): {vis_total.sum() / max(sum(cuenta.values()), 1):.1f}")
    print(f"  Revisa {DESTINO / 'muestras.jpg'} y después: python tools/entrenar_puntos.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
