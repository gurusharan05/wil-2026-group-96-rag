"""Ingest all seven universities + Australia-wide sources into ChromaDB.

Run from the project root:

    PYTHONPATH=. python scripts/ingest_clean_v2.py

What changed in v2.1 (see CHANGES.md):
- University labels come from config.UNIVERSITY_REGISTRY (folder name ->
  canonical name). The previous version only recognised rmit / unimelb /
  monash, so ACU, Deakin, La Trobe and Swinburne documents were stored with
  university="Australia" and could never be filtered by university.
- One canonical label per university (CSV and PDFs used to disagree:
  "The University of Melbourne" vs "University of Melbourne").
- Line-aware chunking: a fact line (e.g. the fee line) is never cut in half.
- The repeated "Important: project snapshot..." disclaimer is moved out of the
  chunk text into metadata so it does not dominate the embeddings.
- Source URL and snapshot date are parsed into metadata for the evidence view.
- The collection records INDEX_VERSION so rag.py can detect a stale index.
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import chromadb
import ollama
from pypdf import PdfReader

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    COURSE_NAMES,
    COURSE_REGISTRY,
    DEFAULT_COURSE_KEY,
    EMBEDDING_MODEL,
    GENERAL_ENTITY,
    INDEX_VERSION,
    OLLAMA_KEEP_ALIVE,
    RAW_DIR,
    STRUCTURED_DIR,
    UNIVERSITY_NAME_FIXES,
    UNIVERSITY_REGISTRY,
)

try:  # BeautifulSoup is only needed if HTML snapshots are added back.
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None


# ============================================================
# Helpers
# ============================================================

def clean_text(text: str) -> str:
    """Clean extracted text."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def canonical_university(name: str) -> str:
    """Map any spelling of a university to its canonical label."""
    name = (name or "").strip()
    return UNIVERSITY_NAME_FIXES.get(name, name)


def infer_metadata(path: Path) -> dict:
    """Infer university and category metadata from a source path.

    The university comes from the folder name (data/raw/<folder>/file.pdf)
    using the shared registry, so every university folder is labelled.
    """
    folder = path.parent.name.lower()
    name = path.stem.lower()

    entry = UNIVERSITY_REGISTRY.get(folder)
    university = entry["name"] if entry else GENERAL_ENTITY

    if "accommodation" in name:
        category = "accommodation"
    elif "course" in name or "master" in name:
        category = "course"
    elif "international" in name:
        category = "international_students"
    elif "fair_work" in name or "home_affairs" in name:
        category = "work_rights"
    else:
        category = "general"

    if "home_affairs" in name:
        authority = "Australian Government — Department of Home Affairs"
    elif "fair_work" in name:
        authority = "Australian Government — Fair Work Ombudsman"
    elif "study_australia" in name:
        authority = "Australian Government — Study Australia"
    elif entry:
        authority = university
    else:
        authority = "Unknown"

    # course.pdf = Master of Data Science; course_<key>.pdf = another course.
    course_key = ""
    if category == "course":
        suffix = name[len("course"):].lstrip("_-") if name.startswith("course") else ""
        course_key = suffix if suffix in COURSE_REGISTRY else DEFAULT_COURSE_KEY

    return {
        "university": university,
        "category": category,
        "course_key": course_key,
        "course_name": COURSE_NAMES.get(course_key, ""),
        "source_type": path.suffix.lower().lstrip("."),
        "doc_type": "document",
        "source_file": f"{folder}/{path.name}",
        "source_url": "",
        "snapshot_date": "",
        "authority": authority,
        "disclaimer": "",
    }


# Lines produced when a web page is "printed to PDF" (La Trobe snapshot).
_PRINT_HEADER = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[AP]M.*|Page \d+ of \d+.*)$"
)


def split_header(text: str) -> tuple[str, dict]:
    """Pull Source URL / Snapshot date / disclaimer out of a snapshot page."""
    lines = [line.rstrip() for line in text.splitlines()]
    info = {"source_url": "", "snapshot_date": "", "disclaimer": ""}
    kept = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if _PRINT_HEADER.match(line):
            i += 1
            continue

        if line.startswith("Source URL:"):
            url = line[len("Source URL:"):].strip()
            # Long URLs wrap onto the next line in the PDF.
            if i + 1 < len(lines) and lines[i + 1].strip() and " " not in lines[i + 1].strip() \
                    and ":" not in lines[i + 1].strip():
                url += lines[i + 1].strip()
                i += 1
            info["source_url"] = url
            kept.append(f"Source URL: {url}")
            i += 1
            continue

        if line.startswith("Snapshot date:"):
            info["snapshot_date"] = line[len("Snapshot date:"):].strip()
            kept.append(line)
            i += 1
            continue

        if line.startswith("Important:"):
            # The disclaimer is a wrapped paragraph that ends with a full
            # stop followed by a short section heading.
            block = [line]
            j = i + 1
            while j < len(lines) and j - i <= 8:
                if block[-1].endswith(".") and len(lines[j].strip()) < 45:
                    break
                block.append(lines[j].strip())
                j += 1
            if block[-1].endswith("."):
                info["disclaimer"] = " ".join(block)
                i = j
                continue

        kept.append(line)
        i += 1

    return clean_text("\n".join(kept)), info


def extract_html(path: Path) -> list[dict]:
    """Extract structured sections from an HTML source."""
    if BeautifulSoup is None:
        raise RuntimeError("beautifulsoup4 is required for HTML sources")

    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    metadata = infer_metadata(path)
    url_tag = soup.find("meta", attrs={"name": "source-url"})
    date_tag = soup.find("meta", attrs={"name": "snapshot-date"})
    if url_tag and url_tag.get("content"):
        metadata["source_url"] = url_tag["content"].strip()
    if date_tag and date_tag.get("content"):
        metadata["snapshot_date"] = date_tag["content"].strip()

    sections = []
    current_title = path.stem
    current_lines = []
    for element in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        text = clean_text(element.get_text(" ", strip=True))
        if not text:
            continue
        if element.name in {"h1", "h2", "h3"}:
            if current_lines:
                sections.append({"section": current_title, "text": "\n".join(current_lines), **metadata})
            current_title = text
            current_lines = []
        else:
            current_lines.append(text)
    if current_lines:
        sections.append({"section": current_title, "text": "\n".join(current_lines), **metadata})
    return sections


def _is_garbage_line(line: str) -> bool:
    """Lines of mangled script/image text from a browser 'print to PDF'."""
    stripped = line.strip()
    if not stripped:
        return False
    symbols = sum(1 for ch in stripped if not (ch.isalnum() or ch.isspace() or ch in ".,;:'’()-–/&$%"))
    return symbols / len(stripped) > 0.2


def _is_navigation_page(text: str) -> bool:
    """True for pages that are mostly menu items / footer links."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 15:
        return False
    short = sum(1 for line in lines if len(line) < 30)
    return short / len(lines) > 0.6


def extract_pdf(path: Path) -> list[dict]:
    """Extract one section per page from a PDF source."""
    reader = PdfReader(str(path))
    base = infer_metadata(path)
    sections = []

    for page_no, page in enumerate(reader.pages, start=1):
        raw = clean_text(page.extract_text() or "")
        text, info = split_header(raw)
        text = "\n".join(line for line in text.splitlines() if not _is_garbage_line(line))

        # Skip pages that are only navigation / print furniture.
        if len(text) < 250 or _is_navigation_page(text):
            continue

        metadata = dict(base)
        for key, value in info.items():
            if value:
                metadata[key] = value
        title = text.splitlines()[0].strip()[:120] if text else path.stem
        sections.append(
            {
                **metadata,
                "section": f"{title} (page {page_no})",
                "page": page_no,
                "text": text,
            }
        )

    # Propagate the header info (only printed on page 1) to later pages.
    if sections:
        for key in ("source_url", "snapshot_date", "disclaimer"):
            first = next((s[key] for s in sections if s.get(key)), "")
            for section in sections:
                if not section.get(key):
                    section[key] = first
    return sections


def _format_fee(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    try:
        return f"{float(value):,.0f}"
    except ValueError:
        return value


def extract_structured_courses(path: Path) -> list[dict]:
    """Convert structured university course facts into RAG-friendly text.

    One record per university. These records are the authoritative source for
    fee / duration / campus / intake questions, and rag.py looks them up
    directly by university (no embedding search needed).
    """
    sections = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            university = canonical_university(row.get("university", ""))
            course_name = row.get("course_name", "").strip()
            course_key = (row.get("course_key", "") or DEFAULT_COURSE_KEY).strip()
            if course_key not in COURSE_REGISTRY:
                print(f"WARNING: unknown course_key {course_key!r} for {university} - {course_name}")
            fee = _format_fee(row.get("international_fee", ""))
            fee_text = f"AUD {fee}" if fee else "Not published in the structured dataset"

            def val(key: str) -> str:
                return (row.get(key, "") or "").strip() or "Not stated"

            text = (
                f"University: {university}\n"
                f"Course: {course_name}\n"
                f"Course code: {val('course_code')}\n"
                f"CRICOS code: {val('cricos_code')}\n"
                f"Field: {val('field')}\n"
                f"Degree level: {val('degree_level')}\n"
                f"Full-time duration: {val('duration_full_time')}\n"
                f"Part-time duration: {val('duration_part_time')}\n"
                f"Advanced standing duration: {val('advanced_standing_duration')}\n"
                f"Study mode: {val('study_mode')}\n"
                f"Campus: {val('campus')}\n"
                f"Published intake: {val('intake')}\n"
                f"International fee: {fee_text}\n"
                f"Fee year: {val('fee_year')}\n"
                f"Fee basis: {val('fee_basis')}\n"
                f"Entry requirements: {val('entry_summary')}\n"
                f"English requirements: {val('english_summary')}\n"
                f"Official course URL: {val('official_course_url')}\n"
                f"Source verified: {val('source_verified')}"
            )
            note = (row.get("verification_note", "") or "").strip()
            if note:
                text += f"\nVerification note: {note}"

            sections.append(
                {
                    "section": f"{university} — {course_name} — Structured Course Facts",
                    "text": text,
                    "university": university,
                    "category": "course",
                    "course_key": course_key,
                    "course_name": course_name,
                    "source_type": "csv",
                    "doc_type": "structured_course",
                    "source_file": path.name,
                    "source_url": row.get("official_course_url", "").strip(),
                    "snapshot_date": row.get("source_verified", "").strip(),
                    "authority": university,
                    "disclaimer": "",
                }
            )
    return sections


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks on line boundaries.

    The old character-based splitter could cut a line such as
    "2026 international tuition fee: AUD 57,984 per EFTSL." in half, so
    neither chunk contained the complete fact.
    """
    if len(text) <= size:
        return [text]

    lines = text.splitlines()
    chunks = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        if current and current_len + len(line) + 1 > size:
            chunks.append("\n".join(current).strip())
            # Carry the last few lines forward as overlap.
            carry: list[str] = []
            carry_len = 0
            for previous in reversed(current):
                if carry_len + len(previous) + 1 > overlap:
                    break
                carry.insert(0, previous)
                carry_len += len(previous) + 1
            current, current_len = carry, carry_len
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def make_id(source_file: str, section: str, chunk_number: int, text: str) -> str:
    """Create a deterministic Chroma document ID."""
    raw = f"{source_file}|{section}|{chunk_number}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_records() -> list[dict]:
    """Read every source and return chunk records ready for embedding."""
    raw_files = sorted(
        p for p in RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".html", ".pdf"}
    )
    print(f"Raw files found: {len(raw_files)}")

    sections = []
    for path in raw_files:
        extracted = extract_html(path) if path.suffix.lower() == ".html" else extract_pdf(path)
        label = extracted[0]["university"] if extracted else "-"
        print(f"Reading: {path.relative_to(RAW_DIR)}  ->  {label}  ({len(extracted)} section(s))")
        sections.extend(extracted)

    unknown_folders = sorted({
        p.parent.name for p in raw_files
        if p.parent.name.lower() not in UNIVERSITY_REGISTRY and p.parent.name.lower() != "australia"
    })
    if unknown_folders:
        print(
            "WARNING: these folders are not in config.UNIVERSITY_REGISTRY and were "
            f"labelled '{GENERAL_ENTITY}': {unknown_folders}"
        )

    structured_course_file = STRUCTURED_DIR / "university_courses.csv"
    if structured_course_file.exists():
        structured_sections = extract_structured_courses(structured_course_file)
        sections.extend(structured_sections)
        print(f"Structured course records added: {len(structured_sections)}")
    else:
        print(f"WARNING: Structured course file not found: {structured_course_file}")

    print(f"Extracted sections: {len(sections)}")

    records = []
    for section in sections:
        # Structured records are short and must never be split.
        pieces = [section["text"]] if section.get("doc_type") == "structured_course" else chunk_text(section["text"])
        for i, text in enumerate(pieces, start=1):
            metadata = {k: v for k, v in section.items() if k != "text"}
            metadata["chunk_id"] = str(i)
            records.append(
                {
                    "id": make_id(section["source_file"], section["section"], i, text),
                    "text": text,
                    "metadata": metadata,
                }
            )
    return records


def main() -> None:
    """Run the complete ingestion pipeline."""
    print("=" * 70)
    print("International Student RAG — ingestion")
    print("=" * 70)

    records = build_records()
    print(f"Created chunks:     {len(records)}")
    print(f"Embedding model:    {EMBEDDING_MODEL}")
    print(f"Collection:         {COLLECTION_NAME}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Removed existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "l2", "index_version": INDEX_VERSION},
    )

    batch_size = 32
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        print(f"Embedding chunks {start + 1}-{start + len(batch)} of {len(records)}...")
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=[record["text"] for record in batch],
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        collection.add(
            ids=[record["id"] for record in batch],
            documents=[record["text"] for record in batch],
            metadatas=[record["metadata"] for record in batch],
            embeddings=response["embeddings"],
        )

    # Summary per university so labelling problems are visible immediately.
    result = collection.get(include=["metadatas"])
    counts: dict = {}
    for metadata in result["metadatas"]:
        key = (metadata.get("university"), metadata.get("category")
               + (f" / {metadata.get('course_key')}" if metadata.get("course_key") else ""))
        counts[key] = counts.get(key, 0) + 1

    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    print(f"Collection: {COLLECTION_NAME} (index_version={INDEX_VERSION})")
    print(f"Documents:  {collection.count()}")
    print(f"Database:   {CHROMA_DIR}")
    print("-" * 70)
    for (university, category), count in sorted(counts.items()):
        print(f"  {university:<38} {category:<24} {count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
