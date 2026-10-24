"""Create course snapshot PDFs from university_courses.csv.

Each non-Data-Science row becomes data/raw/<uni>/course_<course_key>.pdf in
the same format as the team's existing snapshots (an HTML page printed to PDF
by Chromium), including the "Project-created research snapshot ... not an
official university publication" disclaimer.

Run this after correcting a row in the CSV, then rebuild the knowledge base:

    pip install playwright && playwright install chromium   # once
    PYTHONPATH=. python scripts/make_course_snapshots.py
    PYTHONPATH=. python scripts/ingest_clean_v2.py

Optional: CHROMIUM_PATH=/path/to/chrome to use an existing browser.
"""
import csv
import html
import os
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

project = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
html_dir = Path(tempfile.mkdtemp(prefix="course_snapshots_"))

FOLDER = {
    "RMIT University": "rmit",
    "The University of Melbourne": "unimelb",
    "Monash University": "monash",
    "Deakin University": "deakin",
    "La Trobe University": "latrobe",
    "Swinburne University of Technology": "swinburn",
    "Australian Catholic University": "acu",
}
SNAPSHOT_DATE = "2026-10-19"  # change when the data is re-checked
IMPORTANT = ("Project-created research snapshot based on the official page linked above. This is not an "
             "official university publication. Facts are paraphrased, some fields are deliberately marked for "
             "verification, and the official page should be checked before advising students or deploying the "
             "assistant.")

CSS = """
body { font-family: 'Liberation Serif', 'Times New Roman', serif; font-size: 11.6pt; color: #000; line-height: 1.25; }
h1 { font-size: 22.5pt; margin: 0 0 14pt 0; }
h2 { font-size: 16pt; margin: 16pt 0 8pt 0; }
p { margin: 0 0 9pt 0; }
ul { margin: 0; padding-left: 22pt; }
li { margin: 0 0 1pt 0; }
a { color: #0645ad; }
"""


def e(text):
    return html.escape(str(text or "").strip())


def known(value):
    value = (value or "").strip()
    return value and value.lower() not in {"not stated", "not specified"}


def build_html(r):
    uni, course = r["university"], r["course_name"]
    fee = r["international_fee"].strip()
    lines_identity = [
        f"University: {uni}",
        f"Course: {course}",
        f"Course code: {r['course_code']}" if known(r["course_code"]) else "Course code: not captured; verify on the official page",
        f"CRICOS code: {r['cricos_code']}" if known(r["cricos_code"]) else "CRICOS code: not captured; verify on the official page",
        f"Location: {r['campus']}",
        f"Study mode: {r['study_mode']}",
    ]
    lines_duration = [f"Standard duration: {r['duration_full_time']} full-time."]
    if known(r["duration_part_time"]):
        lines_duration.append(f"Part-time: {r['duration_part_time']}.")
    if known(r["advanced_standing_duration"]):
        lines_duration.append(f"Advanced standing: {r['advanced_standing_duration']}.")
    lines_duration.append(f"Published intakes: {r['intake']}.")

    if fee:
        amount = f"{float(fee):,.0f}"
        year = r["fee_year"].strip()
        label = f"{year} international fee" if known(year) else "International fee"
        lines_fee = [f"{label}: AUD {amount}.", f"Fee basis: {r['fee_basis']}."]
    else:
        lines_fee = ["The international fee was not published on the sources available when this snapshot was "
                     "prepared; check the official course page.",
                     f"Note: {r['fee_basis']}."]
    lines_fee.append("Fee values are year-specific and should not be treated as a permanent price.")

    lines_entry = [r["entry_summary"], f"English requirement: {r['english_summary']}"]
    note = r.get("verification_note", "").replace(
        "Collected from official/government web pages with AI assistance on 2026-10-19; team to verify.", "").strip()
    lines_verify = ["Collected from official university and Australian Government web pages on 2026-10-19 "
                    "with AI assistance; to be verified by the team against the official page."]
    if note:
        lines_verify.append(note)

    def ul(items):
        return "<ul>" + "".join(f"<li>{e(i)}</li>" for i in items if str(i).strip()) + "</ul>"

    url = e(r["official_course_url"])
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{e(uni)} — {e(course)}</title>
<style>{CSS}</style></head><body>
<h1>{e(uni)} — {e(course)}</h1>
<p><b>Entity:</b> {e(uni)}</p>
<p><b>Snapshot date:</b> {SNAPSHOT_DATE}</p>
<p><b>Source URL:</b> <a href="{url}">{url}</a></p>
<p><b>Important:</b> {e(IMPORTANT)}</p>
<h2>Course identity</h2>{ul(lines_identity)}
<h2>Duration and intake</h2>{ul(lines_duration)}
<h2>International fee</h2>{ul(lines_fee)}
<h2>Entry requirements</h2>{ul(lines_entry)}
<h2>Verification notes</h2>{ul(lines_verify)}
</body></html>"""


rows = list(csv.DictReader(open(project / "data/structured/university_courses.csv", encoding="utf-8-sig")))
rows = [r for r in rows if r.get("course_key") and r["course_key"] != "data_science"]

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None)
    page = browser.new_page()
    for r in rows:
        folder = FOLDER[r["university"]]
        name = f"course_{r['course_key']}"
        html_path = html_dir / f"{folder}_{name}.html"
        html_path.write_text(build_html(r), encoding="utf-8")
        page.goto(html_path.as_uri())
        out = project / "data/raw" / folder / f"{name}.pdf"
        page.pdf(path=str(out), format="A4",
                 margin={"top": "14mm", "bottom": "12mm", "left": "18mm", "right": "18mm"})
        print("wrote", out.relative_to(project))
    browser.close()
