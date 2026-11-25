"""Adapta examples/soccer/main.py de roboflow/sports a supervision >= 0.31.

1. sv.ByteTrack ya no existe: se sustituye por trackers.ByteTrackTracker.
2. El tracker nuevo marca con -1 las detecciones que aún no tienen track
   (por ejemplo, en los primeros fotogramas). Se filtran, y si en un
   fotograma no queda ninguna se muestra tal cual en vez de fallar.

Se puede ejecutar varias veces: solo aplica los cambios que falten.

Uso (desde la carpeta sports/examples/soccer):
    python <ruta_del_repo>/tools/arreglar_bytetrack_roboflow.py
"""
from pathlib import Path

p = Path("main.py")
original = p.read_text(encoding="utf-8")
s = original

FILTRO = "        detections = detections[detections.tracker_id != -1]\n"
GUARDA = ("        if len(detections) == 0:\n"
          "            yield frame.copy()\n"
          "            continue\n")

# Paso 1: cambiar sv.ByteTrack por trackers.ByteTrackTracker
if "ByteTrackTracker" not in s:
    s = s.replace("import supervision as sv\n",
                  "import supervision as sv\nfrom trackers import ByteTrackTracker\n", 1)
    s = s.replace("tracker = sv.ByteTrack(minimum_consecutive_frames=3)",
                  "tracker = ByteTrackTracker(minimum_consecutive_frames=3, "
                  "track_activation_threshold=0.25, frame_rate=25)")
    s = s.replace("        detections = tracker.update_with_detections(detections)\n",
                  "        detections = tracker.update(detections)\n" + FILTRO)

# Paso 2: no fallar en fotogramas sin ningún jugador con track
if GUARDA not in s:
    s = s.replace(FILTRO, FILTRO + GUARDA)

if s == original:
    print("main.py ya estaba arreglado: no hay nada que cambiar")
else:
    Path("main.py.bak").write_text(original, encoding="utf-8")
    p.write_text(s, encoding="utf-8")
    print("main.py arreglado (copia de la versión anterior en main.py.bak)")
