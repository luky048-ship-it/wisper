import json
import os
import re
import sys
import time
import urllib.request
from difflib import SequenceMatcher

path = sys.argv[1]

API_URL = os.environ.get("LLM_URL", "https://api.provod.ai/v1/chat/completions")
API_KEY = os.environ.get("LLM_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "mimo-v2.6-flash")

if not API_KEY:
    sys.exit("Задай ключ: export LLM_KEY=sk_...")

BATCH = 200
EXTRA = 30
RETRIES = 3
SIM_THRESHOLD = 0.82

SYSTEM = (
    "Ты аналитик русскоязычных технических лекций. Отвечай ТОЛЬКО на русском. "
    "Ниже полный текст лекции с пронумерованными строками. "
    "Сделай сухую выжимку ТОЛЬКО по указанному диапазону строк. "
    "Полный текст дан для контекста, но пункты и понятия формируй "
    "строго по своему диапазону.\n\n"
    "Верни JSON-объект с двумя полями:\n"
    "- sut: массив из 2-5 коротких пунктов сути (строки).\n"
    "- ponyatiya: массив объектов {term, def}, где term — значимое "
    "понятие или термин, реально звучавший в диапазоне, "
    "def — пояснение в 3-7 слов. Не включай общеупотребительные слова "
    "(модель, данные, задача, лекция, занятие, курс, пример, метод, система). "
    "Технические термины (API, endpoint, AI-safety, fine-tuning) "
    "оставляй как в оригинале. Не повторяй понятия из блока 'Уже собрано'."
)


def ask(system, user):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    last_err = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(API_URL, body, headers)
            with urllib.request.urlopen(req, timeout=600) as r:
                full = json.load(r)
            ch = full["choices"][0]
            content = ch["message"]["content"]
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", content, re.DOTALL)
                data = json.loads(m.group(0)) if m else {}
            usage = full.get("usage", {}) or {}
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            print(
                f"  usage: prompt={usage.get('prompt_tokens')} "
                f"completion={usage.get('completion_tokens')} cached={cached}",
                flush=True,
            )
            return data
        except Exception as e:
            last_err = e
            print(f"  ошибка ({e}), попытка {attempt + 1}/{RETRIES}", flush=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"сервер не ответил: {last_err}")


# ---------- чтение входа ----------

lines = []
for raw in open(path, encoding="utf-8"):
    raw = raw.strip()
    if not raw:
        continue
    m = re.match(r"(\[\d\d:\d\d:\d\d\])\s*(.*)", raw)
    if m and m.group(2):
        lines.append((m.group(1), m.group(2)))
    else:
        lines.append(("[--:--:--]", raw))

if not lines:
    sys.exit(f"Не нашёл ни одной строки в {path}")

FULL_TEXT = "\n".join(f"{i + 1}. {t}" for i, (_, t) in enumerate(lines))

STABLE_PREFIX = (
    "ПОЛНЫЙ ТЕКСТ ЛЕКЦИИ:\n"
    + FULL_TEXT
    + "\n\n"
    + "=" * 60
    + "\n"
    + "Сделай выжимку по указанному ниже диапазону строк. "
    + "Суть и понятия — ТОЛЬКО по этому диапазону.\n"
)


def build_user_prompt(start, end, seen_terms):
    parts = [STABLE_PREFIX]
    parts.append(f"\nДИАПАЗОН: строки {start}-{end}\n")
    if seen_terms:
        parts.append("УЖЕ СОБРАНЫ (не повторяй эти понятия):")
        parts.append(", ".join(seen_terms[-80:]))
        parts.append("")
    parts.append("Верни JSON с полями sut и ponyatiya.")
    return "\n".join(parts)


def pick_end(i):
    n = len(lines)
    default = min(i + BATCH, n)
    lo = i + max(1, BATCH // 2)
    hi = min(i + BATCH + EXTRA, n)
    cands = [
        e
        for e in range(lo, hi + 1)
        if lines[e - 1][1].rstrip().endswith((".", "?", "!", "…"))
    ]
    chosen = min(cands, key=lambda e: abs(e - default)) if cands else default

    # Не оставляй короткий хвост: если после chosen осталось меньше половины
    # батча — забираем всё до конца файла.
    if n - chosen < BATCH // 2:
        return n
    return chosen


def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(
        r"(ами|ями|ов|ей|ый|ая|ое|ые|ий|ия|ии|ем|ом|ах|ях|у|ю|и|ы|а|я|е|о)$",
        "",
        s,
    )
    return s.strip()


def similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def add_unique(store, seen, candidate):
    cand = candidate.strip()
    if not cand:
        return
    key = normalize(cand)
    if not key or key in seen:
        return
    for existing in store:
        if similar(existing, cand) >= SIM_THRESHOLD:
            seen.add(key)
            return
    seen.add(key)
    store.append(cand)


def render_ponyatiya(ponyatiya):
    rows = []
    for key, info in sorted(ponyatiya.items(), key=lambda x: -x[1]["count"]):
        best_def = max(info["defs"].items(), key=lambda x: x[1])[0]
        rows.append(f"- **{info['term']}** — {best_def}")
    return "\n".join(rows)


# ---------- основной проход ----------

ponyatiya = {}
sut_points = []
sut_seen = set()

i = 0
while i < len(lines):
    end = pick_end(i)
    seen_terms = [info["term"] for info in ponyatiya.values()]
    user = build_user_prompt(i + 1, end, seen_terms)
    data = ask(SYSTEM, user)

    for p in data.get("sut", []) or []:
        add_unique(sut_points, sut_seen, p)

    for item in data.get("ponyatiya", []) or []:
        term = (item.get("term") or "").strip()
        definition = (item.get("def") or "").strip()
        if not term or not definition:
            continue
        key = normalize(term)
        if not key:
            continue
        if key not in ponyatiya:
            ponyatiya[key] = {"term": term, "defs": {}, "count": 0}
        ponyatiya[key]["count"] += 1
        ponyatiya[key]["defs"][definition] = (
            ponyatiya[key]["defs"].get(definition, 0) + 1
        )

    print(
        f"{end}/{len(lines)}  sut={len(data.get('sut', []) or [])} "
        f"ponyatiya={len(data.get('ponyatiya', []) or [])}",
        flush=True,
    )
    i = end

# ---------- запись ----------

final_sut = "\n".join(f"- {p}" for p in sut_points) or "- (модель не вернула суть)"

out_path = path.replace(".txt", "_summary.md")
with open(out_path, "w", encoding="utf-8") as out:
    out.write("# Выжимка\n\n")
    out.write("## Суть\n\n" + final_sut.strip() + "\n\n")
    out.write("## Понятия\n\n" + (render_ponyatiya(ponyatiya) or "—") + "\n")

print(f"Готово: {out_path}")
