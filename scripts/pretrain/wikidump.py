import os, re, bz2, requests
import mwparserfromhell
from bs4 import BeautifulSoup
from collections import Counter
from datasets import Dataset
from libzim.reader import Archive
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------
DEBUG        = True
MAX_ARTICLES = None #10    # set to None to process all

NOISE_FR = re.compile(
    r"^(une maintenance|cet article est|si tu cherches|cette page d.homonymie"
    r"|améliore(-le)?|à (illustrer|sourcer|compléter)|tu veux peut-être)",
    re.IGNORECASE,
)

SKIP_TITLES   = {"null", "undefined", ""}
ERROR_MARKERS = ("is not available inside this ZIM", "was deleted after")
CATEGORY_RE   = re.compile(r"\[\[Category:([^\]|]+)", re.IGNORECASE)
SKIP_PREFIXES = ("{", "|", "!", "=", "thumb", "right", "left", "center", "File:", "Image:")


# -------------------------
# DOWNLOAD
# -------------------------
def download(url: str, path: str, min_mb: int = 5) -> None:
    if os.path.exists(path) and os.path.getsize(path) / 1e6 > min_mb:
        print(f"[skip] {path}"); return
    r = requests.get(url, stream=True)
    total = int(r.headers.get("Content-Length", 0))
    with open(path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=path) as bar:
        for chunk in r.iter_content(65536):
            f.write(chunk); bar.update(len(chunk))


# -------------------------
# SIMPLE-ENGLISH: XML dump
# -------------------------
def parse_simplewiki(xml_bz2: str) -> list[dict]:
    data = []
    with bz2.open(xml_bz2, "rt", encoding="utf-8", errors="ignore") as f:
        title = text = None
        for line in tqdm(f, desc="simple-english", unit="line"):
            if "<title>" in line:
                m = re.search(r"<title>(.+?)</title>", line)
                title = m.group(1).strip() if m else None
            elif "<text" in line:
                text = re.sub(r"<text[^>]*>", "", line)
            elif text is not None:
                text += line
            if text is not None and "</text>" in line:
                if title and ":" not in title:
                    record = _wikitext_to_record(title, text)
                    if record:
                        data.append(record)
                        if DEBUG and len(data) <= 3:
                            print(f"  [en #{len(data)}] prompt={record['prompt'][:100]!r}")
                            print(f"              text={record['text'][:120]!r}")
                        if MAX_ARTICLES and len(data) >= MAX_ARTICLES:
                            break
                text = None
    print(f"[simple-english] kept={len(data)}")
    return data


def _wikitext_to_record(title: str, raw: str) -> dict | None:
    cats = [c.strip().replace("_", " ") for c in CATEGORY_RE.findall(raw)]

    try:
        wikicode = mwparserfromhell.parse(raw)
        # Remove <ref>...</ref> content before stripping
        for tag in wikicode.filter_tags():
            if tag.tag.strip_code().lower() == "ref":
                try:
                    wikicode.remove(tag)
                except Exception:
                    pass
        text = wikicode.strip_code()
    except Exception:
        return None

    # Remove any remaining HTML tags and refs
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)

    lines = []
    for l in text.splitlines():
        s = l.strip()
        if "Category:" in s:
            break
        if len(s) >= 20 and not s.startswith(SKIP_PREFIXES):
            lines.append(s)
    if not lines:
        return None

    summary = " ".join(lines)#[:600]
    prompt  = f"{title} | {', '.join(cats)}" if cats else title
    return {"prompt": prompt, "text": summary, "seed_data": "simple-english"}


# -------------------------
# VIKIDIA-FR: ZIM
# -------------------------
def parse_vikidia(zim_path: str) -> list[dict]:
    data, skipped = [], 0
    archive = Archive(zim_path)

    with tqdm(total=archive.entry_count, desc="vikidia-fr", unit="entry") as bar:
        for i in range(archive.entry_count):
            bar.set_postfix(kept=len(data), skipped=skipped)
            try:
                entry = archive._get_entry_by_id(i)
                if entry.is_redirect: skipped += 1; bar.update(1); continue
                item = entry.get_item()
                if "html" not in item.mimetype: skipped += 1; bar.update(1); continue
                title   = entry.title.strip()
                content = bytes(item.content).decode("utf-8", errors="ignore")
            except Exception:
                skipped += 1; bar.update(1); continue

            if title.lower() in SKIP_TITLES: skipped += 1; bar.update(1); continue
            if '<meta http-equiv="refresh"' in content: skipped += 1; bar.update(1); continue
            if any(m in content for m in ERROR_MARKERS): skipped += 1; bar.update(1); continue

            record = _html_to_record(title, content)
            if record:
                data.append(record)
                if DEBUG and len(data) <= 3:
                    print(f"  [fr #{len(data)}] prompt={record['prompt'][:100]!r}")
                    print(f"             text={record['text'][:120]!r}")
                if MAX_ARTICLES and len(data) >= MAX_ARTICLES:
                    break
            else:
                skipped += 1
            bar.update(1)

    print(f"[vikidia-fr] kept={len(data)} skipped={skipped}")
    return data


def _html_to_record(title: str, html: str) -> dict | None:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return None

    for tag in soup.find_all(["script", "style", "sup", "figure", "nav", "footer", "table"]):
        tag.decompose()
    for tag in soup.find_all(id=re.compile(r"^(toc|mw-navigation|mw-head|contentSub|catlinks)$", re.I)):
        tag.decompose()

    cats = []
    for a in soup.find_all("a", href=True):
        if "Portail" in a["href"]:
            label = re.sub(r"^Portail\s+(de\s+l['']\s*|des?\s+|du\s+|d['']\s*)?",
                           "", a.get_text(strip=True), flags=re.IGNORECASE).strip()
            if label and label not in cats:
                cats.append(label)

    paragraphs = []
    for p in soup.find_all("p"):
        text = re.sub(r" {2,}", " ", p.get_text(" ", strip=True).replace("\xa0", " "))
        if len(text) < 40 or NOISE_FR.match(text):
            continue
        paragraphs.append(text)
        if len(" ".join(paragraphs)) >= 600:
            break

    if not paragraphs:
        return None

    prompt = f"{title} | {', '.join(cats)}" if cats else title
    return {"prompt": prompt, "text": " ".join(paragraphs)[:600], "seed_data": "vikidia-fr"}


# -------------------------
# RUN
# -------------------------
SIMPLEWIKI_XML = "simplewiki-articles.xml.bz2"
VIKIDIA_ZIM    = "vikidia.zim"

download("https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2", SIMPLEWIKI_XML)
download("https://lb.download.kiwix.org/zim/vikidia/vikidia_fr_all_nopic_2026-05.zim", VIKIDIA_ZIM)

all_data = parse_simplewiki(SIMPLEWIKI_XML) + parse_vikidia(VIKIDIA_ZIM)

dataset = Dataset.from_list(all_data)
print(f"\n{dataset}")
print(f"[distribution] {dict(Counter(dataset['seed_data']))}")
no_cat = sum(1 for r in all_data if "|" not in r["prompt"])
print(f"[no categories] {no_cat}/{len(all_data)} ({no_cat/len(all_data)*100:.1f}%)")

dataset.to_json("simple-wiki.jsonl")
# dataset.push_to_hub("ffurfaro/simple-wiki")
print("[done]")