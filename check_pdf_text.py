from pathlib import Path
from pypdf import PdfReader

files = [
    "data/raw/unimelb/course.pdf",
    "data/raw/rmit/accommodation.pdf",
    "data/raw/unimelb/accommodation.pdf",
    "data/raw/australia/fair_work.pdf",
    "data/raw/monash/course.pdf",
]

terms = [
    "57984", "rental", "tenancy", "budgeting",
    "residential colleges", "workplace rights", "Clayton"
]

for filename in files:
    path = Path(filename)
    print(f"\n--- {filename} ---")

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    text = "\n".join(
        page.extract_text() or ""
        for page in PdfReader(str(path)).pages
    )

    print(f"Extracted characters: {len(text)}")

    for term in terms:
        start = 0
        while True:
            pos = text.lower().find(term.lower(), start)
            if pos == -1:
                break

            excerpt = text[max(0, pos - 100):pos + 250]
            print(f"\nMatch for {term!r}:")
            print(" ".join(excerpt.split()))
            break
