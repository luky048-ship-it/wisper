import json
import re
import sys
import time
import urllib.request
from difflib import SequenceMatcher

path = sys.argv[1]

URL = "http://localhost:8080/v1/chat/completions"

BATCH = 40
EXTRA = 10
RETRIES = 3
SIM_THRESHOLD = 0.82
DEBUG = True

SYSTEM = (
    "Ты аналитик русскоязычных технических лекций. Отвечай ТОЛЬКО на русском. "
    "могут быть обрывками фраз. Сделай сухую выжимку: только суть. "
    "Никаких вводных слов, пояснений, мета-комментариев.\n\n"
    "Формат ответа СТРОГО такой, больше ничего:\n"
    "СУТЬ:\n"
    "- пункт один\n"
    "- пункт два\n"
    "ТЕРМИНЫ:\n"
    "- термин :: пояснение в 3-7 слов\n"
    "- термин :: пояснение в 3-7 слов\n\n"
    "Правила:\n"
    "1. СУТЬ — каждый с новой строки, начинается с '- '.\n"
    "2. ТЕРМИНЫ — только специфические понятия, реально звучавшие тут. "
    "Не включай общеупотребительные слова (модель, данные, задача, "
    "лекция, занятие, курс, пример, метод, система).\n"
    "3. Каждый термин — на своей строке, формат: '- термин :: пояснение'.\n"
    "4. Никаких других секций. Никаких ТЕМЫ, ДОПОЛНИТЕЛЬНО, ВЫВОДЫ.\n"
    "5. Если термин уже есть в блоке 'Уже собраны', НЕ повторяй его."
)

FINAL_SYSTEM = (
    "Ты собираешь итоговую суть русскоязычной лекции. Отвечай только на русском. "
    "На вход — пункты сути по частям одной лекции. "
    "Сделай 5-10 связных пунктов без дублей и без воды. "
    "Ничего не выдумывай, только то, что есть во входных данных. "
    "Формат ответа строго:\n"
    "СУТЬ:\n- ...\n- ..."
)


def ask(system, user):
    body = json.dumps(
        {
            "messages": [
                {"role": "system", "content": system},
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
        except Exception as e:
            last_err = e
            print(f"  ошибка ({e}), попытка {attempt + 1}/{RETRIES}", flush=True)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"сервер не ответил: {last_err}")


def build_user_prompt(texts, seen_terms):
    parts = []
    if seen_terms:
        parts.append("Уже собраны (НЕ повторяй эти термины):")
        parts.append(", ".join(seen_terms[-60:]))
        parts.append("")
    parts.append("Строки для выжимки:")
    parts.extend(f"{k + 1}. {t}" for k, t in enumerate(texts))
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
    return min(cands, key=lambda e: abs(e - default)) if cands else default


# ---------- парсинг ----------

SECTION_RE = re.compile(r"^\s*(СУТЬ|ТЕРМИНЫ)\s*:?\s*$", re.IGNORECASE)


def normalize_sections(text):
    text = text.replace("**", "").replace("*", "")
    text = re.sub(
        r"\s*(СУТЬ|ТЕРМИНЫ)\s*:",
        r"\n\1:\n",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def split_sections(text):
    out = {"СУТЬ": [], "ТЕРМИНЫ": []}
    cur = None
    text = normalize_sections(text)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = SECTION_RE.match(line)
        if m:
            cur = m.group(1).upper()
            continue
        # игнорируем всё, что не СУТЬ и не ТЕРМИНЫ
        if cur not in ("СУТЬ", "ТЕРМИНЫ"):
            continue
        body = re.sub(r"^\s*[-•*]\s*", "", line).strip()
        if not body:
            continue
        if cur == "СУТЬ":
            out["СУТЬ"].append(body)
        else:
            for sep in (" :: ", "::", " — ", "—", " – ", "–"):
                if sep in body:
                    term, _, definition = body.partition(sep)
                    term = term.strip()
                    definition = definition.strip()
                    if term and definition:
                        out["ТЕРМИНЫ"].append((term, definition))
                    break
    return out


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


STOPWORDS = {
    "модель",
    "данные",
    "задача",
    "лекция",
    "занятие",
    "курс",
    "пример",
    "метод",
    "система",
    "подход",
    "вопрос",
    "ответ",
}


def add_term(ponyatiya, term, definition):
    key = normalize(term)
    if not key or key in STOPWORDS:
        return
    if key not in ponyatiya:
        ponyatiya[key] = {"term": term, "defs": {}, "count": 0}
    ponyatiya[key]["count"] += 1
    ponyatiya[key]["defs"][definition] = ponyatiya[key]["defs"].get(definition, 0) + 1


def render_ponyatiya(ponyatiya):
    rows = []
    for key, info in sorted(ponyatiya.items(), key=lambda x: -x[1]["count"]):
        best_def = max(info["defs"].items(), key=lambda x: x[1])[0]
        rows.append(f"- **{info['term']}** — {best_def}")
    return "\n".join(rows)


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

# ---------- основной проход ----------

ponyatiya = {}
sut_points = []
sut_seen = set()


def process_batch(batch):
    seen = [info["term"] for info in ponyatiya.values()]
    ans, _ = ask(SYSTEM, build_user_prompt([t for _, t in batch], seen))
    sections = split_sections(ans)
    if not sections["СУТЬ"] and not sections["ТЕРМИНЫ"] and len(batch) > 1:
        half = batch[: max(1, len(batch) // 2)]
        ans, _ = ask(SYSTEM, build_user_prompt([t for _, t in half], seen))
        sections = split_sections(ans)
    return sections


i = 0
while i < len(lines):
    end = pick_end(i)
    batch = lines[i:end]
    sections = process_batch(batch)

    if DEBUG:
        print(
            f"  [{i}:{end}] суть={len(sections['СУТЬ'])} "
            f"термины={len(sections['ТЕРМИНЫ'])}",
            flush=True,
        )

    for p in sections["СУТЬ"]:
        add_unique(sut_points, sut_seen, p)
    for term, definition in sections["ТЕРМИНЫ"]:
        add_term(ponyatiya, term, definition)

    i = end
    print(f"{i}/{len(lines)}", flush=True)

# ---------- финальная суть ----------

joined = "\n".join(f"- {p}" for p in sut_points)
final_sut = ""
if joined.strip():
    raw_final, _ = ask(FINAL_SYSTEM, joined)
    fs = split_sections(raw_final)
    if fs["СУТЬ"]:
        # дедуп на всякий случай
        final_seen = set()
        final_list = []
        for p in fs["СУТЬ"]:
            add_unique(final_list, final_seen, p)
        final_sut = "\n".join(f"- {p}" for p in final_list)
    else:
        final_sut = joined

if not final_sut.strip():
    final_sut = "- (модель не вернула суть)"

# ---------- запись ----------

out_path = path.replace(".txt", "_summary.md")
with open(out_path, "w", encoding="utf-8") as out:
    out.write("# Выжимка\n\n")
    out.write("## Суть\n\n" + final_sut.strip() + "\n\n")
    out.write("## Понятия\n\n" + (render_ponyatiya(ponyatiya) or "—") + "\n")

print(f"Готово: {out_path}")
