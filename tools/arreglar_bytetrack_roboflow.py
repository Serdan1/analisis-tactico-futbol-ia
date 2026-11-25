"""Adapta examples/soccer/main.py de roboflow/sports a supervision >= 0.31,
donde sv.ByteTrack ya no existe (ahora está en la librería `trackers`).
Uso (desde la carpeta sports/examples/soccer):
    python <ruta_del_repo>/tools/arreglar_bytetrack_roboflow.py
"""
from pathlib import Path

p = Path("main.py")
s = p.read_text(encoding="utf-8")
if "ByteTrackTracker" in s:
    print("main.py ya estaba arreglado")
    raise SystemExit
s = s.replace("import supervision as sv\n",
              "import supervision as sv\nfrom trackers import ByteTrackTracker\n", 1)
s = s.replace("tracker = sv.ByteTrack(minimum_consecutive_frames=3)",
              "tracker = ByteTrackTracker(minimum_consecutive_frames=3, track_activation_threshold=0.25, frame_rate=25)")
s = s.replace("detections = tracker.update_with_detections(detections)",
              "detections = tracker.update(detections)\n        detections = detections[detections.tracker_id != -1]")
p.with_suffix(".py.bak").write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
p.write_text(s, encoding="utf-8")
print("main.py arreglado (copia del original en main.py.bak)")
