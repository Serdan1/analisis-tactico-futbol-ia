"""Pasar nuestro sistema por clips de SoccerNet GSR y compararlo con la verdad (tarea #25).

    python tools/evaluar_soccernet.py --n 10                 # los 10 primeros clips de test
    python tools/evaluar_soccernet.py --clips SNGS-116 SNGS-130
    python tools/evaluar_soccernet.py --n 10 --solo_evaluar  # sin volver a pasar YOLO

Para cada clip: imágenes -> data/clips/SNGS-XXX.mp4 -> main.py hasta "equipos"
(con los parámetros de siempre, sin tocar nada) -> evaluación. Al final, una
tabla en outputs/soccernet_resultados.csv. Cada clip tarda ~4-5 min en la GPU;
lo ya calculado se reutiliza, así que se puede parar (Ctrl+C) y seguir otro día.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as sistema                                   # noqa: E402
from src.evaluacion_soccernet import evaluar_clip         # noqa: E402

ORIGEN = Path("data/soccernet/SN-GSR-2025")


def a_mp4(clip: str) -> Path:
    destino = Path("data/clips") / f"{clip}.mp4"
    if destino.exists():
        return destino
    imgs = sorted((ORIGEN / clip / "img1").glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"No hay imágenes en {ORIGEN / clip / 'img1'}")
    h, w = cv2.imread(str(imgs[0])).shape[:2]
    destino.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    for p in imgs:
        out.write(cv2.imread(str(p)))
    out.release()
    return destino


def fila_resumen(clip, r):
    t, eq, sg = r.get("tracking", {}), r.get("equipos", {}), r.get("seguimiento", {})
    return dict(clip=clip, detectadas_pct=r["detectadas_pct"], con_posicion_pct=r["con_posicion_pct"],
                error_mediana_m=r["error_mediana_m"], error_p95_m=r["error_p95_m"],
                error_menos_1m_pct=r["error_menos_1m_pct"], cambios_numero=t.get("cambios_de_numero"),
                cambios_por_jugador=t.get("cambios_por_jugador"),
                equipos_acierto_pct=eq.get("acierto_pct"), equipos_colores=" / ".join(eq.get("colores", {})),
                sentido_ok=eq.get("sentido_correcto"),
                reid_bien_pct=sg.get("reid", {}).get("bien_pct"), reid_otro_pct=sg.get("reid", {}).get("otro_pct"),
                tracker_bien_pct=sg.get("solo_tracker", {}).get("bien_pct"),
                tracker_otro_pct=sg.get("solo_tracker", {}).get("otro_pct"),
                jugadores_seguidos=sg.get("reid", {}).get("jugadores"), transformacion=r["transformacion"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", help="nombres SNGS-XXX (por defecto, todos los descargados)")
    ap.add_argument("--n", type=int, help="solo los n primeros")
    ap.add_argument("--solo_evaluar", action="store_true", help="no ejecutar main.py: solo evaluar lo que ya hay")
    ap.add_argument("--sin_reid", action="store_true", help="no evaluar el seguimiento de cada jugador (más rápido)")
    ap.add_argument("--max_jugadores", type=int, default=None, help="seguir como mucho N jugadores por clip")
    a = ap.parse_args()

    clips = a.clips or sorted(p.name for p in ORIGEN.glob("SNGS-*") if (p / "Labels-GameState.json").exists())
    clips = clips[:a.n] if a.n else clips
    if not clips:
        print(f"✗ No hay clips en {ORIGEN}. Ejecuta antes tools/descargar_soccernet.py"); return 1

    filas = []
    for k, clip in enumerate(clips, 1):
        print(f"\n===== [{k}/{len(clips)}] {clip} =====", flush=True)
        t0 = time.time()
        try:
            video = a_mp4(clip)
            if not a.solo_evaluar:
                if sistema.main(["--video", str(video).replace("\\", "/"), "--hasta", "equipos", "--silencio"]) != 0:
                    print("  ✗ main.py ha fallado en este clip: se salta"); continue
            r = evaluar_clip(Path("outputs") / clip, ORIGEN / clip / "Labels-GameState.json",
                             con_reid=not a.sin_reid, max_jugadores=a.max_jugadores)
        except KeyboardInterrupt:
            print("\nParado. Lo calculado está guardado: vuelve a ejecutar para seguir."); break
        except Exception as e:                       # un clip raro no para todo
            print(f"  ✗ {type(e).__name__}: {e}"); continue
        f = fila_resumen(clip, r); filas.append(f)
        print(f"  error mediano {f['error_mediana_m']} m · detectadas {f['detectadas_pct']} % · "
              f"cambios de número {f['cambios_numero']} · equipos {f['equipos_acierto_pct']} % · "
              f"seguimiento bien: re-ID {f['reid_bien_pct']} % / solo tracker {f['tracker_bien_pct']} % "
              f"({time.time() - t0:.0f} s)")

    if filas:
        df = pd.DataFrame(filas)
        Path("outputs").mkdir(exist_ok=True)
        df.to_csv("outputs/soccernet_resultados.csv", index=False)
        print("\n" + df[["clip", "error_mediana_m", "error_p95_m", "detectadas_pct", "cambios_numero",
                         "equipos_acierto_pct", "sentido_ok", "reid_bien_pct", "reid_otro_pct",
                         "tracker_bien_pct", "tracker_otro_pct"]].to_string(index=False))
        print("\nGuardado en outputs/soccernet_resultados.csv (y outputs/<clip>/evaluacion.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
