
"""Crawl all chapter subpages of the selected Bengali Wikisource edition."""
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

API_URL = "https://bn.wikisource.org/w/api.php"
BOOK_TITLE = "মেজদিদি (শরৎচন্দ্র চট্টোপাধ্যায়, ১৯৬৩)"
BOOK_URL = "https://bn.wikisource.org/wiki/" + quote(BOOK_TITLE.replace(" ", "_"))
OUTPUT = Path("data/book.json")
HEADERS = {
    "User-Agent": "BanglaBookRAGStudentProject/1.0 (educational project)"
}
session = requests.Session()
session.headers.update(HEADERS)


def api_get(params):
    response = session.get(API_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload


def get_all_chapter_titles():
    """Use MediaWiki allpages continuation to discover every subpage."""
    prefix = BOOK_TITLE + "/"
    titles = []
    continuation = {}

    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "allpages",
            "apnamespace": 0,
            "apprefix": prefix,
            "aplimit": "max",
        }
        params.update(continuation)
        payload = api_get(params)

        for page in payload.get("query", {}).get("allpages", []):
            title = page["title"]
            # Chapter pages have exactly: root / section / chapter
            suffix = title[len(prefix):] if title.startswith(prefix) else ""
            parts = suffix.split("/")
            if len(parts) == 2 and all(parts):
                titles.append(title)

        continuation = payload.get("continue")
        if not continuation:
            break

    return sorted(set(titles))


def page_html(title):
    payload = api_get({
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "text",
    })
    return payload["parse"]["text"]["*"]


def clean_page(html):
    soup = BeautifulSoup(html, "lxml")
    content = soup.select_one("#mw-content-text") or soup.select_one(".mw-parser-output")
    if content is None:
        return ""

    for node in content.select(
        "script, style, table, .navbox, .catlinks, .mw-editsection, "
        ".noprint, .metadata, .ws-noexport, .printfooter, .mw-jump-link"
    ):
        node.decompose()

    text = content.get_text(separator="\n", strip=True)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def crawl():
    print("Selected Wikisource edition:")
    print(BOOK_URL)
    print("\nDiscovering chapter subpages...")
    titles = get_all_chapter_titles()

    if not titles:
        raise RuntimeError(
            "No chapter pages were discovered. Check the book title or Wikisource API."
        )

    print(f"Discovered {len(titles)} chapter pages.")
    records = []

    for index, title in enumerate(titles, start=1):
        try:
            html = page_html(title)
            text = clean_page(html)
            if len(text) < 100:
                print(f"[SKIP] Short/empty page: {title}")
                continue

            suffix = title[len(BOOK_TITLE) + 1:]
            section, chapter = suffix.split("/", 1)
            url = "https://bn.wikisource.org/wiki/" + quote(title.replace(" ", "_"))

            records.append({
                "book": "মেজদিদি",
                "edition": "১৯৬৩",
                "author": "শরৎচন্দ্র চট্টোপাধ্যায়",
                "section": section,
                "chapter": chapter,
                "title": title,
                "source_url": url,
                "text": text,
            })
            print(f"[{index}/{len(titles)}] Downloaded: {section} / {chapter}")
            time.sleep(0.25)
        except Exception as exc:
            print(f"[ERROR] {title}: {exc}")

    if not records:
        raise RuntimeError("No usable chapter text was downloaded.")

    # Sort by the order shown in the edition's table of contents.
    section_order = {"মেজদিদি": 0, "আঁধারে আলো": 1}
    chapter_order = {
        "এক": 1, "দুই": 2, "তিন": 3, "চার": 4,
        "পাঁচ": 5, "ছয়": 6, "ছয়": 6, "সাত": 7, "আট": 8
    }
    records.sort(key=lambda r: (
        section_order.get(r["section"], 99),
        chapter_order.get(r["chapter"], 99),
        r["chapter"]
    ))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\nCrawl complete.")
    print(f"Usable chapter pages: {len(records)}")
    print(f"Saved: {OUTPUT.resolve()}")
    print("Expected edition structure: 14 chapters (8 + 6).")
    if len(records) != 14:
        print("WARNING: Expected 14 pages. Inspect the downloaded titles and API results.")


if __name__ == "__main__":
    crawl()
