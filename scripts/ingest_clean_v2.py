"""Ingest the three-university + Australia-wide dataset into ChromaDB."""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import chromadb
import ollama
from bs4 import BeautifulSoup
from pypdf import PdfReader

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    RAW_DIR,
    STRUCTURED_DIR,
)


def clean_text(text: str) -> str:
    """Clean extracted text."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_metadata(path: Path) -> dict:
    """Infer university and category metadata from a source path."""
    parts = {p.lower() for p in path.parts}
    name = path.stem.lower()

    if "rmit" in parts or "rmit" in name:
        university = "RMIT University"
    elif "unimelb" in parts or "melbourne" in parts:
        university = "University of Melbourne"
    elif "monash" in parts or "monash" in name:
        university = "Monash University"
    else:
        university = "Australia"

    if "accommodation" in name:
        category = "accommodation"
    elif "course" in name or "master" in name:
        category = "course"
    elif "international_students" in name or "international" in name:
        category = "international_students"
    elif "fair_work" in name:
        category = "work_rights"
    elif "home_affairs" in name:
        category = "work_rights"
    elif "study_australia" in name:
        category = "general"
    else:
        category = "general"

    if "home_affairs" in name:
        authority = "Australian Government — Department of Home Affairs"
    elif "fair_work" in name:
        authority = "Australian Government — Fair Work Ombudsman"
    elif "study_australia" in name:
        authority = "Australian Government — Study Australia"
    elif university != "Australia":
        authority = university
    else:
        authority = "Unknown"

    return {
        "university": university,
        "category": category,
        "source_type": path.suffix.lower().lstrip("."),
        "source_file": path.name,
        "source_url": "",
        "snapshot_date": "",
        "authority": authority,
    }


def extract_html(path: Path) -> list[dict]:
    """Extract structured sections from an HTML source."""
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(
        ["script", "style", "nav", "footer", "header", "noscript"]
    ):
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
                sections.append(
                    {
                        "section": current_title,
                        "text": "\n".join(current_lines),
                        **metadata,
                    }
                )

            current_title = text
            current_lines = []
        else:
            current_lines.append(text)

    if current_lines:
        sections.append(
            {
                "section": current_title,
                "text": "\n".join(current_lines),
                **metadata,
            }
        )

    return sections


def extract_pdf(path: Path) -> list[dict]:
    """Extract one section per page from a PDF source."""
    reader = PdfReader(str(path))
    metadata = infer_metadata(path)
    sections = []

    for page_no, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")

        if text:
            sections.append(
                {
                    "section": f"Page {page_no}",
                    "page": page_no,
                    "text": text,
                    **metadata,
                }
            )

    return sections


def extract_structured_courses(path: Path) -> list[dict]:
    """
    Convert structured university course facts into RAG-friendly text.

    The structured course CSV contains important facts such as:
    - course code
    - CRICOS code
    - duration
    - campus
    - intake
    - international fee
    - fee basis
    - entry requirements
    - English requirements
    """

    sections = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = csv.DictReader(f)

        for row in rows:
            university = row.get("university", "").strip()
            course_name = row.get("course_name", "").strip()

            text = (
                f"University: {university}\n"
                f"Course: {course_name}\n"
                f"Course code: {row.get('course_code', '').strip()}\n"
                f"CRICOS code: {row.get('cricos_code', '').strip()}\n"
                f"Field: {row.get('field', '').strip()}\n"
                f"Degree level: {row.get('degree_level', '').strip()}\n"
                f"Full-time duration: "
                f"{row.get('duration_full_time', '').strip()}\n"
                f"Part-time duration: "
                f"{row.get('duration_part_time', '').strip()}\n"
                f"Advanced standing duration: "
                f"{row.get('advanced_standing_duration', '').strip()}\n"
                f"Study mode: {row.get('study_mode', '').strip()}\n"
                f"Campus: {row.get('campus', '').strip()}\n"
                f"Published intake: {row.get('intake', '').strip()}\n"
                f"International fee: "
                f"AUD {row.get('international_fee', '').strip()}\n"
                f"Fee year: {row.get('fee_year', '').strip()}\n"
                f"Fee basis: {row.get('fee_basis', '').strip()}\n"
                f"Entry requirements: "
                f"{row.get('entry_summary', '').strip()}\n"
                f"English requirements: "
                f"{row.get('english_summary', '').strip()}\n"
                f"Official course URL: "
                f"{row.get('official_course_url', '').strip()}\n"
                f"Source verified: "
                f"{row.get('source_verified', '').strip()}"
            )

            sections.append(
                {
                    "section": (
                        f"{university} — "
                        f"{course_name} — Structured Course Facts"
                    ),
                    "text": text,
                    "university": university,
                    "category": "course",
                    "source_type": "csv",
                    "source_file": path.name,
                    "source_url": row.get(
                        "official_course_url",
                        "",
                    ).strip(),
                    "snapshot_date": row.get(
                        "source_verified",
                        "",
                    ).strip(),
                    "authority": university,
                }
            )

    return sections


def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split long text into overlapping chunks."""
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(
            end - overlap,
            start + 1,
        )

    return chunks


def make_id(
    source_file: str,
    section: str,
    chunk_number: int,
    text: str,
) -> str:
    """Create a deterministic Chroma document ID."""
    raw = (
        f"{source_file}|"
        f"{section}|"
        f"{chunk_number}|"
        f"{text}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def main() -> None:
    """Run the complete ingestion pipeline."""

    print("=" * 70)
    print("International Student RAG — ingestion")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Find raw HTML and PDF files
    # ---------------------------------------------------------

    raw_files = sorted(
        [
            p
            for p in RAW_DIR.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".html", ".pdf"}
        ]
    )

    print(f"Raw files found: {len(raw_files)}")

    sections = []

    for path in raw_files:
        print(
            f"Reading: "
            f"{path.relative_to(RAW_DIR)}"
        )

        if path.suffix.lower() == ".html":
            sections.extend(
                extract_html(path)
            )
        else:
            sections.extend(
                extract_pdf(path)
            )

    # ---------------------------------------------------------
    # 2. Add structured university course data
    # ---------------------------------------------------------

    structured_course_file = (
        STRUCTURED_DIR / "university_courses.csv"
    )

    if structured_course_file.exists():
        print(
            f"Reading structured course data: "
            f"{structured_course_file}"
        )

        structured_sections = (
            extract_structured_courses(
                structured_course_file
            )
        )

        sections.extend(
            structured_sections
        )

        print(
            "Structured course records added: "
            f"{len(structured_sections)}"
        )
    else:
        print(
            "WARNING: Structured course file not found: "
            f"{structured_course_file}"
        )

    print(
        f"Extracted sections: {len(sections)}"
    )

    # ---------------------------------------------------------
    # 3. Create chunks
    # ---------------------------------------------------------

    records = []

    for section in sections:
        for i, text in enumerate(
            chunk_text(section["text"]),
            start=1,
        ):
            metadata = {
                k: v
                for k, v in section.items()
                if k != "text"
            }

            metadata["chunk_id"] = f"{i}"

            records.append(
                {
                    "id": make_id(
                        section["source_file"],
                        section["section"],
                        i,
                        text,
                    ),
                    "text": text,
                    "metadata": metadata,
                }
            )

    print(
        f"Created chunks:     {len(records)}"
    )

    # ---------------------------------------------------------
    # 4. Connect to ChromaDB
    # ---------------------------------------------------------

    print(
        f"Embedding model:    {EMBEDDING_MODEL}"
    )

    print(
        f"Collection:         {COLLECTION_NAME}"
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # ---------------------------------------------------------
    # 5. Rebuild collection
    # ---------------------------------------------------------

    try:
        client.delete_collection(
            COLLECTION_NAME
        )

        print(
            f"Removed existing collection: "
            f"{COLLECTION_NAME}"
        )

    except Exception:
        pass

    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "l2"},
    )

    # ---------------------------------------------------------
    # 6. Generate embeddings and add documents
    # ---------------------------------------------------------

    batch_size = 32

    for start in range(
        0,
        len(records),
        batch_size,
    ):
        batch = records[
            start:start + batch_size
        ]

        print(
            f"Embedding chunks "
            f"{start + 1}-"
            f"{start + len(batch)} "
            f"of {len(records)}..."
        )

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=[
                record["text"]
                for record in batch
            ],
        )

        collection.add(
            ids=[
                record["id"]
                for record in batch
            ],
            documents=[
                record["text"]
                for record in batch
            ],
            metadatas=[
                record["metadata"]
                for record in batch
            ],
            embeddings=response["embeddings"],
        )

    # ---------------------------------------------------------
    # 7. Completion message
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        f"Documents:  {collection.count()}"
    )

    print(
        f"Database:   {CHROMA_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()