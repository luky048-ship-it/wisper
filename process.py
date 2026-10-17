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
import subprocess
import numpy as np
from faster_whisper import WhisperModel

path = sys.argv[1]
lang = sys.argv[2] if len(sys.argv) > 2 else "en"  # "en" или "ru"
base = os.path.splitext(path)[0]


def load_audio(p):
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            p,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def fmt(sec):
    sec = int(sec)
    return f"[{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}]"


# ---- 1. распознавание ----
print("Распознаю...")
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
segments, _ = model.transcribe(
    load_audio(path),
    language=lang,
    beam_size=5,
    vad_filter=True,
    initial_prompt=(
        "Hello, and welcome. Let's get started with today's lecture."
        if lang == "en"
        else "Здравствуйте. Начинаем сегодняшнюю лекцию."
    ),
    no_speech_threshold=0.6,
    compression_ratio_threshold=2.4,
)

# собираем сегменты в целые предложения
MAX_CHARS = 220


def split_long(text):
    parts = []
    while len(text) > MAX_CHARS:
        cut = text.rfind(", ", 80, MAX_CHARS)
        if cut == -1:
            cut = text.rfind(" ", 80, MAX_CHARS)
        if cut == -1:
            cut = MAX_CHARS
        parts.append(text[: cut + 1].strip())
        text = text[cut + 1 :].strip()
    parts.append(text)
    return parts


sentences = []  # (время начала, текст)


def add(ts, text):
    for sent in re.split(r"(?<=[.!?])\s+", text):
        for p in split_long(sent):
            if p:
                sentences.append((ts, p))


cur, cur_ts = "", 0.0
for s in segments:
    text = s.text.strip()
    if not re.search(r"[^\W\d_]", text):
        continue
    if cur and not re.search(r"[.!?]$", cur) and len(cur) + len(text) < MAX_CHARS:
        cur += " " + text  # мысль не закончена, клеим
    else:
        if cur:
            add(cur_ts, cur)
        cur, cur_ts = text, s.start
if cur:
    add(cur_ts, cur)


with open(base + ".txt", "w", encoding="utf-8") as f:
    for ts, s in sentences:
        f.write(f"{fmt(ts)} {s}\n")
print(f"Оригинал: {base}.txt ({len(sentences)} предложений)")

del model

# ---- 2. перевод (только для английского) ----
if lang != "ru":
    import ctranslate2, transformers

    print("Перевожу...")
    translator = ctranslate2.Translator(
        "nllb-1300m-ct2", device="cpu", compute_type="int8", intra_threads=6
    )
    tok = transformers.AutoTokenizer.from_pretrained(
        "facebook/nllb-200-distilled-1.3B", src_lang="eng_Latn"
    )
    with open(base + "_ru.txt", "w", encoding="utf-8") as f:
        for i in range(0, len(sentences), 16):
            batch = sentences[i : i + 16]
            toks = [tok.convert_ids_to_tokens(tok.encode(s)) for _, s in batch]
            res = translator.translate_batch(
                toks,
                target_prefix=[["rus_Cyrl"]] * len(toks),
                beam_size=4,
                max_decoding_length=256,
            )
            for (ts, _), r in zip(batch, res):
                ru = tok.decode(tok.convert_tokens_to_ids(r.hypotheses[0][1:]))
                f.write(f"{fmt(ts)} {ru}\n")
            print(f"  {min(i + 16, len(sentences))}/{len(sentences)}", flush=True)
    print(f"Перевод: {base}_ru.txt")
