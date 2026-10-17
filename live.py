import ctypes, os, sys
import nvidia.cublas.lib, nvidia.cudnn.lib

for pkg in (nvidia.cublas.lib, nvidia.cudnn.lib):
    d = list(pkg.__path__)[0]
    for f in sorted(os.listdir(d)):
        if ".so" in f and "nvblas" not in f:
            try:
                ctypes.CDLL(os.path.join(d, f), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass

import re
import queue
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import ctranslate2
import transformers
from faster_whisper import WhisperModel

# ---- настройки ----
MONITOR = "alsa_output.pci-0000_07_00.4.analog-stereo.monitor"
SRC_LANG = sys.argv[1] if len(sys.argv) > 1 else "en"  # "en" или "ru"
MODEL_SIZE = "medium"
DEVICE = "cuda"
COMPUTE_TYPE = "float16" if MODEL_SIZE == "small" else "int8_float16"
CHUNK_SEC = 4
MAX_BUFFER_CHARS = 200

TRANSLATE = SRC_LANG != "ru"
NLLB_DIR = "nllb-600m-ct2"
NLLB_TOKENIZER = "facebook/nllb-200-distilled-600M"
# -------------------

RATE = 16000
CHUNK_BYTES = RATE * 2 * CHUNK_SEC
SAVE = False

# единственный файл на выходе: оригинальный текст
out = None
if SAVE:
    Path("transcripts").mkdir(exist_ok=True)
    out_path = (
        Path("transcripts") / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{SRC_LANG}.txt"
    )
    out = open(out_path, "a", encoding="utf-8")

if TRANSLATE:
    print("Загружаю NLLB...")
    translator = ctranslate2.Translator(
        NLLB_DIR, device="cpu", compute_type="int8", intra_threads=4
    )
    tok = transformers.AutoTokenizer.from_pretrained(
        NLLB_TOKENIZER, src_lang="eng_Latn"
    )

    def translate(text):
        tokens = tok.convert_ids_to_tokens(tok.encode(text))
        res = translator.translate_batch(
            [tokens],
            target_prefix=[["rus_Cyrl"]],
            beam_size=4,
            max_decoding_length=256,
        )
        return tok.decode(tok.convert_tokens_to_ids(res[0].hypotheses[0][1:]))

    translate("Hello world.")  # прогрев


def clean(text):
    text = re.sub(r"\.{2,}|…", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if re.search(r"[^\W\d_]", text) else ""


def fmt(sec):
    sec = int(sec)
    return f"[{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}]"


print("Загружаю Whisper...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

audio_q = queue.Queue()
sent_q = queue.Queue()
stop = threading.Event()

proc = subprocess.Popen(
    [
        "parec",
        "-d",
        MONITOR,
        "--format=s16le",
        f"--rate={RATE}",
        "--channels=1",
        "--latency-msec=100",
    ],
    stdout=subprocess.PIPE,
)
assert proc.stdout is not None
stream = proc.stdout


def capture_thread():
    n = 0
    while not stop.is_set():
        raw = stream.read(CHUNK_BYTES)
        if not raw:
            break
        audio_q.put((n * CHUNK_SEC, raw))
        n += 1


def translate_thread():
    while not stop.is_set() or not sent_q.empty():
        try:
            ts, sentence = sent_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if TRANSLATE:
            print(f"{ts} {sentence}\n      → {translate(sentence)}\n", flush=True)
        else:
            print(f"{ts} {sentence}\n", flush=True)


def emit(ts, sentence):
    if out:
        out.write(f"{ts} {sentence}\n")
        out.flush()
    sent_q.put((ts, sentence))


threading.Thread(target=capture_thread, daemon=True).start()
tr = threading.Thread(target=translate_thread)
tr.start()
print("Готово, слушаю...\n")

buffer, buffer_ts = "", ""
try:
    while True:
        t_start, raw = audio_q.get()
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = model.transcribe(
            audio,
            vad_filter=True,
            beam_size=1,
            language=SRC_LANG,
            condition_on_previous_text=False,
        )
        text = clean(" ".join(s.text.strip() for s in segments))

        if not text:  # пауза: выводим накопленное
            if buffer:
                emit(buffer_ts, buffer)
                buffer = ""
            continue

        if buffer:
            # точка на границе чанка, а дальше маленькая буква: фраза не закончена
            if buffer[-1] in ".!?" and text[0].islower():
                buffer = buffer[:-1]
            buffer += " " + text
        else:
            buffer, buffer_ts = text, fmt(t_start)

        parts = re.split(r"(?<=[.!?])\s+", buffer)
        for s in parts[:-1]:
            emit(buffer_ts, s)
            buffer_ts = fmt(t_start)
        buffer = parts[-1]

        if len(buffer) > MAX_BUFFER_CHARS:
            emit(buffer_ts, buffer)
            buffer = ""
except KeyboardInterrupt:
    pass
finally:
    if buffer:
        emit(buffer_ts, buffer)
    stop.set()
    proc.terminate()
    tr.join(timeout=15)  # дать переводу дописать
    if out:
        out.close()
