"""main.py debe fallar con mensajes claros, no con trazas de Python."""
import main


def test_video_que_no_existe(capsys):
    assert main.main(["--video", "data/clips/no_existe.mp4", "--track_id", "7"]) == 1
    assert "No encuentro el vídeo" in capsys.readouterr().out
