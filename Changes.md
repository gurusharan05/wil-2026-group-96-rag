# Fixes: slow answers, wrong fees, unrelated evidence

This file explains **why** each reported problem happened and **how** it was fixed.

> **Required after pulling these changes:** rebuild the knowledge base once.
>
> ```bash
> PYTHONPATH=. python scripts/ingest_clean_v2.py
> ```
>
> The old `data/chroma_db` was built by the old ingestion script and stores the
> wrong university labels. `rag.py` now detects an old index and shows this
> command instead of silently giving wrong answers.

---

## Problem 1: Slow responses, even for "what is the RMIT fee?"

### Why it happened

1. **Qwen3's "thinking" mode was never actually turned off.**
   `generate_answer()` passed `"think": False` inside `options={...}`. In the
   Ollama Python client, `think` is a **top-level argument** of `ollama.chat()`.
   Ollama ignores unknown keys inside `options`, so `qwen3:14b` wrote a long
   hidden reasoning trace before every answer. That trace is often hundreds to
   thousands of tokens, and it was the biggest cause of the slowness.
2. **Simple fee questions often reached the LLM anyway.** The fast path only
   read `chunks[0]` and only recognised a few exact line prefixes. Melbourne's
   page says `2026 international tuition fee: …`, La Trobe's says
   `Annual tuition fee A$45 600 …`, and neither prefix was in the list. Any
   miss meant a full 14B-model generation.
3. **Substring keyword matching sent fee questions down the wrong path.**
   `"rent" in "current"`, `"work" in "coursework"` and `"fee" in "coffee"` are
   all `True`. So *"What are the current fees at RMIT?"* was treated as an
   accommodation question too, which disabled the fast path and called the LLM.
4. **The vector DB was reopened and fully scanned on every question.**
   `get_mentioned_universities()` created a new `PersistentClient` and read the
   metadata of every chunk to discover university names, then `retrieve()`
   opened the DB again.
5. **Models were unloaded between questions.** Ollama unloads an idle model
   after 5 minutes by default, so the next question paid the full load time
   for the 14B model again.

### How it was fixed

| Fix | Where |
|---|---|
| `think=False` passed as a top-level `ollama.chat()` argument, with a fallback to qwen3's `/no_think` switch on older Ollama versions; any `<think>` block is stripped | `rag.py → _chat()` |
| Fact questions (fee, duration, campus, intake, course code, CRICOS, entry, English) are answered **directly from the structured course record** of the named university, with **no embedding call and no LLM call** | `rag.py → retrieve()`, `_deterministic_answer()` |
| Word-boundary regexes replace substring checks for topics and facts | `rag.py → FACT_PATTERNS`, `*_PATTERN` |
| Cached Chroma client/collection; universities detected from a fixed registry instead of a DB scan; query embeddings cached | `rag.py → _client()`, `get_mentioned_universities()`, `_embed_cached()` |
| `keep_alive="30m"` on every Ollama call, and the app warms both models once at startup | `config.OLLAMA_KEEP_ALIVE`, `rag.warm_up()`, `app.py` |
| LLM output capped (`num_predict=600`, `num_ctx=4096`) and **streamed** into the UI so text appears immediately | `config.py`, `rag._stream_tokens()`, `app.py` |
| The UI shows how long each answer took and whether the LLM was used | `app.py` |

Result: a fee / duration / campus question now takes milliseconds (zero model
calls). Only open-ended questions ("What will I study at RMIT?") use the LLM,
and those now run without the hidden thinking pass.

---

## Problem 2: Wrong fee answers for some universities

### Why it happened

1. **The ingestion script only knew 3 of the 7 universities.**
   `infer_metadata()` checked only for `rmit`, `unimelb` and `monash`. All
   **ACU, Deakin, La Trobe and Swinburne** PDFs fell through to
   `university = "Australia"`, the same label as the government sources.
   (The `data/chroma_db` in the zip shows this: Deakin, La Trobe, Swinburne
   and ACU course pages are all stored as `Australia / course`.)
2. **So questions about those universities were never filtered.** University
   detection read the labels from the DB. With no "Deakin University" label
   (or only the CSV row), retrieval either ran with **no university filter**
   or missed the PDF pages, and the top chunk could belong to any university.
   The fast path then read the fee from `chunks[0]`, which could be another
   university's fee.
3. **Alias gaps.** Aliases were derived automatically from the label, so
   "Swinburne University of Technology" never got the alias `swinburne`,
   La Trobe never got `latrobe`, and so on.
4. **Two labels for Melbourne.** The CSV said `The University of Melbourne`
   and the PDFs said `University of Melbourne`. A filter on one label missed
   the other half of the data.
5. **La Trobe's fee was blank in `university_courses.csv`,** so the fast path
   returned `International Fee: AUD` with no number, even though
   `data/raw/latrobe/course.pdf` states *"Annual tuition fee A$45 600 per 120
   credit points"*.
6. **Character-based chunking could cut a fact line in half,** so neither chunk
   held the complete fee sentence.

### How it was fixed

| Fix | Where |
|---|---|
| One **university registry** (folder → canonical name → aliases) shared by ingestion and retrieval | `config.UNIVERSITY_REGISTRY` |
| Ingestion labels every file by its folder through the registry, normalises CSV names, and prints a per-university summary so labelling mistakes are visible | `scripts/ingest_clean_v2.py` |
| Fee answers come from the structured record **of the named university only**, and the source-page sentence that states the fee is shown alongside it for corroboration | `rag.py` |
| La Trobe row filled from the La Trobe PDF already in the project (AUD 45,600, 2027, per 120 credit points, CRICOS 092396B, Bundoora, March/July/November) | `data/structured/university_courses.csv` |
| Line-aware chunking (a line is never split); structured records are never split | `chunk_text()` |
| Disclaimer paragraph moved out of chunk text into metadata; navigation-only pages of the La Trobe web print are skipped | `split_header()`, `_is_navigation_page()` |
| The index stores `index_version`; `rag.py` refuses an old index with a clear rebuild message | `config.INDEX_VERSION`, `rag._cached_collection()` |

---

## Problem 3: Evidence shows other universities for a single-university question

### Why it happened

1. **"Australia" was treated as a university name.** It was in the DB labels,
   so any question containing "Australia" (e.g. *"RMIT fees for international
   students in Australia"*) also searched the "Australia" bucket, which held
   **Deakin, La Trobe, Swinburne and ACU** documents because of the mislabelling
   above.
2. **Unrecognised universities meant no filter at all**, so the search ran
   over the whole collection (see Problem 2).
3. **Topic false positives added extra searches.** For example, "current"
   triggered an accommodation search.
4. **The UI listed every retrieved chunk as "Evidence used"**, even when the
   answer came from only one of them.

### How it was fixed

* Every search is restricted to the universities named in the question, and
  "Australia" is only used for government work-rights sources.
* Fact questions return at most **two** evidence items, both from the
  university asked about: the structured course record plus the page that
  states the fact.
* Comparisons ("compare RMIT and Monash", "which university is cheapest?")
  return one structured record per university that was asked about.
* `answer_question()` returns the answer **together with the evidence actually
  used**, and the app shows only that, with the official URL and snapshot date.

---

## How the pipeline works now

```
question
  └─ plan_query()        universities (registry aliases), topics, facts
       └─ retrieve()
            ├─ fact question + university   → metadata lookup (no embedding)
            ├─ comparison                   → structured record per university
            ├─ accommodation + university   → that university's pages
            ├─ work rights                  → government sources only
            └─ open question                → vector search filtered by university/topic
       └─ answer
            ├─ structured/deterministic     → instant, no LLM
            └─ otherwise                    → qwen3 (think=False), streamed
```

## Verifying

```bash
python -m pytest tests/ -v                            # 43 offline tests, no Ollama needed
PYTHONPATH=. python scripts/evaluate_retrieval.py     # retrieval metrics (needs Ollama)
```

The tests cover: alias detection for all 7 universities, the substring bugs,
correct fees for every university, single-university evidence, the
`think=False` call, streaming, and ingestion labels.

## Known data gaps (not code bugs)

The accommodation snapshots for **Deakin, La Trobe, ACU and Swinburne** only
point to the official page. They do not contain options such as "homestay",
"private rentals" or "Rental Advisory Service". Evaluation questions Q33–Q35
expect those phrases, so they will keep failing until richer snapshots are
added to `data/raw/<uni>/accommodation.pdf` and the index is rebuilt. The
Swinburne course PDF also has no fee; its fee (AUD 45,010, 2026) comes from
the structured CSV.

## Compatibility notes

* `retrieve()`, `generate_answer()`, `RetrievedChunk`, `get_collection()`,
  `detect_university()` and `get_mentioned_universities()` keep their names,
  so `scripts/evaluate_retrieval.py` and `scripts/test_retrieval.py` work
  unchanged.
* Chunks found by metadata lookup (not by vector search) have
  `distance = 0.0` and `metadata["_retrieval"] = "direct"`.
* Canonical Melbourne label: **The University of Melbourne** (matches the CSV
  and `evaluation_questions.csv`).


---

## Extension: more courses (19 Oct 2026)

**What was added**

* 26 new course records in `data/structured/university_courses.csv` for Master of Information Technology (7 universities), Cyber Security (7), Artificial Intelligence (6) and Business Analytics (6). Total: 33 course records.
* Two new CSV columns: `course_key` (links a row to `config.COURSE_REGISTRY`) and `verification_note` (how the row was collected and what to check).
* `config.COURSE_REGISTRY`: course names and nicknames ("MIT", "IT", "cyber", "AI", "business analytics"). "IT"/"AI" only count in capitals, so "is it online?" is not treated as an IT question.
* Retrieval and answers filter by course as well as by university:
  * "RMIT Master of IT fee" → that one course record.
  * "What are the fees for RMIT?" (no course named) → a table of all RMIT courses.
  * "Compare cyber security fees at Monash and Swinburne" → one table, cyber only.
  * "Which universities offer a Master of AI?" → table, plus which universities have no record.
  * "Does Swinburne have a Master of Business Analytics?" → says there is no record, never invents one.
* Compare Courses tab: course selector, "no record" note, verification badge, link to the official page.
* `INDEX_VERSION` is now 3, so the knowledge base must be rebuilt once.
* Tests: 67 (24 new) cover course detection, every new course's fee and evidence, missing courses and unpublished fees.

**How the data was collected:** from official university pages and guides (RMIT course pages, the University of Melbourne 2026 international fee table and course pages, Monash's 2027 International Graduate Course Guide, Deakin and Swinburne course pages) and, where university pages could not be read, the Australian Government's Study Australia course search (La Trobe, ACU). Each new row has a `verification_note`, and the app and answers mark these rows until the team clears the note.

**Things the team should check**

* **La Trobe (all four new courses):** fees are blank. La Trobe's pages load with JavaScript and Study Australia shows "Enquire". Save each course page as a PDF from a browser (as was done for Data Science) and fill in the fee.
* **RMIT:** pages show AU$47,040 (international) and AU$38,400 (likely domestic) in different places; confirm. MC271 CRICOS needs checking.
* **Monash:** fees are the 2026 quoted fees from the 2027 international guide; Cyber Security and Business Analytics CRICOS codes could not be read reliably.
* **Online-only / non-Melbourne courses:** University of Melbourne and ACU Cyber Security and AI are online only (no course CRICOS, so not student-visa courses); Deakin Cyber Security and Applied AI are on campus at Waurn Ponds (Geelong); ACU Master of IT is at North Sydney. These are labelled in the data.
* **Not found:** Swinburne Master of AI; ACU Master of Business Analytics (ACU only has a Graduate Certificate in Data Analytics). Swinburne's Business Analytics entry is the Master of Business Information Systems (Business Analytics specialisation), the closest equivalent.
* **Snapshot PDFs:** each new course has `data/raw/<uni>/course_<key>.pdf` (26 files) in the same format as the existing snapshots, generated from the CSV by `scripts/make_course_snapshots.py` and carrying the same "project-created, not an official publication" disclaimer. After correcting a CSV row, re-run that script and then the ingestion script so the PDF and CSV stay in step. You can also replace a generated PDF with a real browser printout of the official page (keep the same file name).
