import json
import re
import sys
import time
import urllib.request

path = sys.argv[1]

URL = "http://localhost:8080/v1/chat/completions"

BATCH = 10  # целевой размер батча (строк)
EXTRA = 6  # на сколько строк батч может вырасти, чтобы попасть на конец предложения
CONTEXT = 5  # сколько уже переведённых строк показываем назад
LOOKAHEAD = 4  # сколько следующих строк показываем вперёд (только для понимания)
RETRIES = 3  # повторы при сетевых ошибках
MIN_RATIO = 0.3  # если перевод короче оригинала более чем втрое, переспрашиваем

END = (".", "?", "!", "…", '."', '?"', '!"')

SYSTEM = (
    "Ты переводчик технических лекций с английского на русский. "
    "Переводи точно, ничего не пропускай и не добавляй. "
    "Технические термины (API, endpoint, frontend, backend и т.п.) "
    "оставляй по-английски или давай общепринятый русский вариант. "
    "Строки являются субтитрами, нарезанными по времени, поэтому могут быть "
    "обрывками предложений. Переводи каждую строку как соответствующий ей "
    "фрагмент, сохраняя порядок, и не переноси слова между строками. "
    "Блок 'Дальше по тексту' нужен только для понимания смысла: "
    "не переводи его и не включай в ответ. "
    "Строки пронумерованы. Верни ровно столько же строк с теми же номерами "
    "в формате '1. текст', без комментариев."
)


def ask(user):
    """Возвращает (текст ответа, finish_reason). Повторяет при сетевых ошибках."""
    body = json.dumps(
        {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
        }
    ).encode()
    last_err = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(
                URL, body, {"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=600) as r:
                ch = json.load(r)["choices"][0]
            return ch["message"]["content"], ch.get("finish_reason")
        except Exception as e:  # сеть, таймаут, 5xx
            last_err = e
            print(
                f"  ошибка запроса ({e}), попытка {attempt + 1}/{RETRIES}", flush=True
            )
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"сервер не ответил: {last_err}")


def parse(answer, n):
    """Разбирает '1. текст'. Переносы внутри перевода склеивает с предыдущим пунктом."""
    items = {}
    cur = None
    for line in answer.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(\d+)\.\s*(.*)$", line)
        if m and 1 <= int(m.group(1)) <= n and m.group(1) not in items:
            cur = m.group(1)
            items[cur] = m.group(2)
        elif cur is not None:
            items[cur] += " " + line
    return [items.get(str(i + 1)) or None for i in range(n)]


def build_prompt(texts, ahead, header="Переведи эти строки:"):
    ctx = ""
    if history:
        ctx = (
            "Контекст (уже переведено, НЕ переводи заново, "
            "но держись тех же терминов):\n"
            + "\n".join(f"EN: {e}\nRU: {r}" for e, r in history[-CONTEXT:])
            + "\n\n"
        )
    fut = ""
    if ahead:
        fut = (
            "\n\nДальше по тексту (ТОЛЬКО для понимания смысла, "
            "НЕ переводи и не включай в ответ):\n" + "\n".join(ahead)
        )
    body = "\n".join(f"{k + 1}. {t}" for k, t in enumerate(texts))
    return f"{ctx}{header}\n{body}{fut}"


def pick_end(i):
    """Конец батча: ближайшая к BATCH строка с концом предложения, иначе просто BATCH."""
    n = len(lines)
    default = min(i + BATCH, n)
    lo = i + max(1, BATCH // 2)
    hi = min(i + BATCH + EXTRA, n)
    cands = [e for e in range(lo, hi + 1) if lines[e - 1][1].rstrip().endswith(END)]
    return min(cands, key=lambda e: abs(e - default)) if cands else default


def translate_single(idx):
    """Перевод одной строки с контекстом назад и вперёд."""
    src = lines[idx][1]
    ahead = [t for _, t in lines[idx + 1 : idx + 1 + LOOKAHEAD]]
    ans, _ = ask(build_prompt([src], ahead, "Переведи эту строку:"))
    return parse(ans, 1)[0]


def suspicious(src, ru):
    return len(src) > 15 and len(ru) < MIN_RATIO * len(src)


lines = []
for raw in open(path, encoding="utf-8"):
    m = re.match(r"(\[\d\d:\d\d:\d\d\])\s*(.*)", raw.strip())
    if m and m.group(2):
        lines.append((m.group(1), m.group(2)))

history = []  # (оригинал, перевод) уже переведённых строк

out_path = path.replace(".txt", "_ru_llm.txt")
with open(out_path, "w", encoding="utf-8") as out:
    i = 0
    while i < len(lines):
        end = pick_end(i)
        batch = lines[i:end]
        ahead = [t for _, t in lines[end : end + LOOKAHEAD]]

        ans, reason = ask(build_prompt([t for _, t in batch], ahead))
        if reason == "length":
            print(
                "  ответ обрезан по max_tokens, недостающие строки уйдут в fallback",
                flush=True,
            )
        res = parse(ans, len(batch))

        for k, (ts, src) in enumerate(batch):
            ru = res[k]
            if ru is None or suspicious(src, ru):
                ru = translate_single(i + k) or ru or f"[НЕ ПЕРЕВЕДЕНО] {src}"
            out.write(f"{ts} {ru}\n")
            history.append((src, ru))
        out.flush()

        i = end
        print(f"{i}/{len(lines)}", flush=True)

print(f"Готово: {out_path}")
