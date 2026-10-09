"""Where knowledge-base documents come from: Wikipedia, or files on disk.

Standard library only. `urllib` fetches, and plain-text files are read
directly, so growing the corpus adds no dependency (Section 4, rule 6).

**Wikipedia through its API, not by scraping pages.** The `extracts` endpoint
returns the article as plain text, with a stable URL and a revision the reader
can check. Scraping rendered HTML would pull in navigation boxes, citations and
table fragments that embed as noise and retrieve as nonsense.

**Reference sections are dropped.** "See also", "References" and the like are
lists of titles, not statements of fact. Left in, they embed close to almost
any query that names a topic, because they are made of nothing but names, and
they would win retrieval slots while containing no answer.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

__all__ = ["fetch_wikipedia", "paragraphs_of", "read_text_file"]

API = "https://en.wikipedia.org/w/api.php"
# Wikimedia asks every client to identify itself; anonymous agents are throttled.
USER_AGENT = "SMART-ANN/0.1 (on-device RAG research project; educational use)"

_HEADING = re.compile(r"^\s*=+\s*(.*?)\s*=+\s*$")
_STOP_SECTIONS = {
    "see also",
    "references",
    "notes",
    "external links",
    "further reading",
    "bibliography",
    "sources",
    "citations",
    "footnotes",
}


def paragraphs_of(text: str) -> list[str]:
    """Split plain text into paragraphs, stopping at the first reference section.

    Headings are removed rather than kept as paragraphs: a line reading
    "History" is not a sentence, and a chunk made of one would match every
    question that mentions history.
    """
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _HEADING.match(line)
        if m:
            if m.group(1).strip().lower() in _STOP_SECTIONS:
                break
            continue
        out.append(line)
    return out


def fetch_wikipedia(title: str, timeout: float = 30.0) -> tuple[str, str, list[str]]:
    """The article's canonical title, its URL, and its paragraphs.

    Follows redirects, so "google ceo" style capitalisation and aliases land on
    the real article. Raises LookupError for a page that does not exist rather
    than returning an empty document that would index as nothing.
    """
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "format": "json",
        "titles": title,
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.URLError as exc:
        raise ConnectionError(f"could not reach Wikipedia for '{title}': {exc}") from exc

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        raise LookupError(f"Wikipedia returned no page for '{title}'")
    page = next(iter(pages.values()))
    if "missing" in page or not page.get("extract"):
        raise LookupError(f"Wikipedia has no article titled '{title}'")

    canonical = str(page.get("title", title))
    page_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(canonical.replace(" ", "_"))
    return canonical, page_url, paragraphs_of(str(page["extract"]))


def read_text_file(path: Path) -> tuple[str, str, list[str]]:
    """A local .txt or .md file as a titled document.

    Markdown headings are treated like Wikipedia's: dropped as text, and a
    reference section ends the document. PDFs are refused with a reason rather
    than half-read, because no PDF reader is installed and adding one would be
    a new dependency.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist")
    if p.suffix.lower() == ".pdf":
        raise ValueError(
            f"{p.name}: PDF is not supported without a PDF library. Save it as .txt "
            f"first, or copy the text into a .md file."
        )
    if p.suffix.lower() not in {".txt", ".md", ".markdown", ".text"}:
        raise ValueError(f"{p.name}: only .txt and .md files are supported")

    text = p.read_text(encoding="utf-8", errors="replace")
    # Markdown headings become the same shape as Wikipedia's so one parser serves both.
    text = re.sub(r"^\s*#{1,6}\s*(.+?)\s*#*\s*$", r"== \1 ==", text, flags=re.MULTILINE)
    return p.stem.replace("_", " "), p.resolve().as_uri(), paragraphs_of(text)
