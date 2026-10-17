import subprocess
from datetime import datetime
from pathlib import Path

MONITOR = "alsa_output.pci-0000_07_00.4.analog-stereo.monitor"

Path("recordings").mkdir(exist_ok=True)
path = Path("recordings") / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.wav"
print(f"Запись в {path}. Остановка: Ctrl+C")

proc = subprocess.Popen(
    [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "pulse",
        "-i",
        MONITOR,
        "-ac",
        "1",
        "-ar",
        "16000",
        str(path),
    ]
)
try:
    proc.wait()
except KeyboardInterrupt:
    proc.wait()  # ffmpeg сам получил Ctrl+C и дописывает файл
print("Сохранено:", path)
