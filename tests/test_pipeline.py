"""Regression tests for the three reported problems.

1. Slow answers        -> simple fact questions make no LLM call; the LLM is
                          called with think=False as a top-level argument.
2. Wrong fee answers   -> every university returns its own fee.
3. Unrelated evidence  -> evidence only comes from the university asked about.

Run:  python -m pytest tests/ -v
"""
from __future__ import annotations

import pytest

EXPECTED_FEES = {
    "RMIT University": ("What are the data science fees for RMIT?", "47,040"),
    "The University of Melbourne": ("What is the data science tuition fee at the University of Melbourne?", "57,984"),
    "Monash University": ("Monash data science fees", "57,540"),
    "Deakin University": ("What is the Master of Data Science fee at Deakin?", "45,800"),
    "La Trobe University": ("How much does the Master of Data Science cost at La Trobe?", "45,600"),
    "Swinburne University of Technology": ("What is the international data science fee for Swinburne?", "45,010"),
    "Australian Catholic University": ("What are ACU data science fees?", "37,788"),
}

# (question, university, course_key, fee that must appear)
NEW_COURSE_FEES = [
    ("What is the fee for the Master of IT at RMIT?", "RMIT University", "it", "47,040"),
    ("RMIT business analytics fee", "RMIT University", "business_analytics", "50,880"),
    ("unimelb master of information technology fees", "The University of Melbourne", "it", "64,000"),
    ("University of Melbourne business analytics tuition", "The University of Melbourne", "business_analytics", "91,488"),
    ("Monash cybersecurity fee", "Monash University", "cyber", "57,300"),
    ("Monash Master of Artificial Intelligence fee", "Monash University", "ai", "57,300"),
    ("Deakin business analytics fee", "Deakin University", "business_analytics", "45,800"),
    ("Swinburne cyber security fee", "Swinburne University of Technology", "cyber", "46,990"),
    ("Swinburne Master of IT fee", "Swinburne University of Technology", "it", "45,910"),
    ("ACU master of information technology fee", "Australian Catholic University", "it", "76,192"),
    ("ACU cyber security fee", "Australian Catholic University", "cyber", "38,164"),
    ("Swinburne business analytics fee", "Swinburne University of Technology", "business_analytics", "40,300"),
    ("ACU Master of Artificial Intelligence fee", "Australian Catholic University", "ai", "37,788"),
]


# ---------------------------------------------------------------- routing

@pytest.mark.parametrize("question, expected", [
    ("rmit fees", ["RMIT University"]),
    ("What does unimelb charge?", ["The University of Melbourne"]),
    ("University of Melbourne tuition", ["The University of Melbourne"]),
    ("swinburne fee", ["Swinburne University of Technology"]),
    ("Swinburne University of Technology fee", ["Swinburne University of Technology"]),
    ("latrobe fees", ["La Trobe University"]),
    ("La Trobe fees", ["La Trobe University"]),
    ("ACU fees", ["Australian Catholic University"]),
    ("Australian Catholic University fees", ["Australian Catholic University"]),
    ("deakin fees", ["Deakin University"]),
    ("compare monash and rmit", ["Monash University", "RMIT University"]),
    # The city / country must not be mistaken for a university.
    ("cost of living in Melbourne, Australia", []),
    ("What are the fees for RMIT students in Australia?", ["RMIT University"]),
])
def test_university_detection(rag_module, question, expected):
    assert rag_module.get_mentioned_universities(question) == expected


@pytest.mark.parametrize("question, topics", [
    ("What are the current fees at RMIT?", ["course"]),           # "current" is not "rent"
    ("RMIT coursework fees", ["course"]),                         # "coursework" is not "work"
    ("Is there a coffee shop? RMIT fee", ["course"]),             # "coffee" is not "fee"
    ("How much is rent near Monash?", ["accommodation"]),
    ("How many hours can I work per fortnight?", ["work_rights"]),
])
def test_topic_detection_uses_word_boundaries(rag_module, question, topics):
    assert rag_module.plan_query(question).topics == topics


# ------------------------------------------------- problem 2: right answer

@pytest.mark.parametrize("university", list(EXPECTED_FEES))
def test_fee_answer_is_correct_for_every_university(rag_module, calls, university):
    question, fee = EXPECTED_FEES[university]
    result = rag_module.answer_question(question)
    assert fee in result.answer, result.answer
    for other, (_, other_fee) in EXPECTED_FEES.items():
        if other != university and other_fee != fee:
            assert other_fee not in result.answer


# ------------------------------------------------- problem 3: evidence

@pytest.mark.parametrize("university", list(EXPECTED_FEES))
def test_fee_evidence_only_from_requested_university(rag_module, university):
    question, _ = EXPECTED_FEES[university]
    result = rag_module.answer_question(question)
    assert result.evidence
    assert {c.metadata["university"] for c in result.evidence} == {university}
    assert len(result.evidence) <= 2


def test_country_word_does_not_pull_other_universities(rag_module):
    result = rag_module.answer_question("What are the current fees for RMIT international students in Australia?")
    assert {c.metadata["university"] for c in result.evidence} == {"RMIT University"}


def test_open_question_is_filtered_to_university(rag_module, calls):
    result = rag_module.answer_question("What will I study in the RMIT Master of Data Science?")
    assert result.route == "llm"
    assert {c.metadata["university"] for c in result.evidence} == {"RMIT University"}


def test_comparison_includes_only_named_universities(rag_module):
    result = rag_module.answer_question("Compare the fees of RMIT and Monash")
    assert {c.metadata["university"] for c in result.evidence} == {"RMIT University", "Monash University"}
    assert "47,040" in result.answer and "57,540" in result.answer


@pytest.mark.parametrize("question", [
    "How many hours can I work per fortnight on a student visa?",
    # Q16 of the evaluation set: "course" here must not trigger course retrieval.
    "How many hours can a student visa holder generally work per fortnight while the course is in session?",
])
def test_work_rights_come_from_government_sources(rag_module, question):
    result = rag_module.answer_question(question)
    assert "48 hours" in result.answer
    assert {c.metadata["university"] for c in result.evidence} == {"Australia"}


# ------------------------------------------------- problem 1: speed

def test_fee_question_needs_no_model_calls(rag_module, calls):
    result = rag_module.answer_question("What are the fees for RMIT?")
    assert result.route == "structured"
    assert calls.embed_calls == 0
    assert calls.chat_calls == []


def test_llm_called_with_thinking_disabled(rag_module, calls):
    result = rag_module.answer_question("What will I study in the RMIT Master of Data Science?")
    assert result.answer == "LLM answer"  # <think> block stripped
    kwargs = calls.chat_calls[-1]
    assert kwargs.get("think") is False                      # top-level argument
    assert "think" not in kwargs.get("options", {})          # not hidden in options
    assert kwargs.get("keep_alive")


def test_streaming_hides_think_block(rag_module, calls):
    result = rag_module.answer_question("What will I study in the RMIT Master of Data Science?", stream=True)
    assert "".join(result.stream) == "LLM answer"


def test_out_of_domain_rejected(rag_module):
    result = rag_module.answer_question("What is the weather in Melbourne tomorrow?")
    assert result.route in {"no_evidence", "llm"}
    assert not result.plan.universities


# ------------------------------------------------- ingestion

def test_every_university_folder_is_labelled(rag_module):
    collection = rag_module.get_collection()
    metadatas = collection.get(include=["metadatas"])["metadatas"]
    labels = {m["source_file"].split("/")[0]: m["university"] for m in metadatas if "/" in m["source_file"]}
    assert labels == {
        "acu": "Australian Catholic University",
        "australia": "Australia",
        "deakin": "Deakin University",
        "latrobe": "La Trobe University",
        "monash": "Monash University",
        "rmit": "RMIT University",
        "swinburn": "Swinburne University of Technology",
        "unimelb": "The University of Melbourne",
    }
    structured = {m["university"] for m in metadatas if m.get("doc_type") == "structured_course"}
    assert len(structured) == 7  # every university has at least one course record


def test_chunking_keeps_lines_whole():
    import ingest_clean_v2 as ingest

    text = "\n".join(f"Line {i}: " + "x" * 80 for i in range(40))
    for chunk in ingest.chunk_text(text, size=400, overlap=100):
        for line in chunk.splitlines():
            assert line.startswith("Line ") and line.endswith("x")


# ------------------------------------------------- multiple courses

@pytest.mark.parametrize("question, expected", [
    ("RMIT master of IT fees", ["it"]),
    ("is it online?", []),                                   # lower-case "it" is not a course
    ("What AI masters does Monash have?", ["ai"]),
    ("cyber security at Deakin", ["cyber"]),
    ("cybersecurity", ["cyber"]),
    ("Master of Business Analytics", ["business_analytics"]),
    ("data science vs information technology", ["data_science", "it"]),
    ("RMIT Master of Business Analytics and AI Strategy", ["business_analytics", "ai"]),
])
def test_course_detection(rag_module, question, expected):
    assert rag_module.detect_courses(question) == expected


@pytest.mark.parametrize("question, university, course_key, fee", NEW_COURSE_FEES)
def test_new_course_fee_and_evidence(rag_module, calls, question, university, course_key, fee):
    result = rag_module.answer_question(question)
    assert fee in result.answer, result.answer
    assert result.evidence
    assert {c.metadata["university"] for c in result.evidence} == {university}
    assert {c.metadata["course_key"] for c in result.evidence} == {course_key}
    assert calls.chat_calls == []


def test_university_without_course_lists_all_its_courses(rag_module):
    result = rag_module.answer_question("What are the fees for RMIT?")
    for name in ["Master of Data Science", "Master of Information Technology", "Master of Cyber Security",
                 "Master of Artificial Intelligence", "Master of Business Analytics and AI Strategy"]:
        assert name in result.answer
    assert {c.metadata["university"] for c in result.evidence} == {"RMIT University"}


def test_missing_course_is_reported_not_invented(rag_module, calls):
    result = rag_module.answer_question("Does ACU have a master of business analytics?")
    assert "No Master of Business Analytics record" in result.answer
    assert calls.chat_calls == []


def test_unpublished_fee_is_not_guessed(rag_module):
    result = rag_module.answer_question("What is the La Trobe Master of Cybersecurity fee?")
    assert "does not specify" in result.answer
    assert "AUD" not in result.answer.split("Fee basis")[0].split("International fee")[-1]


def test_course_comparison_only_named_universities_and_course(rag_module):
    result = rag_module.answer_question("Compare cyber security fees at Monash and Swinburne")
    assert {c.metadata["university"] for c in result.evidence} == {"Monash University", "Swinburne University of Technology"}
    assert {c.metadata["course_key"] for c in result.evidence} == {"cyber"}
    assert "57,300" in result.answer and "46,990" in result.answer


def test_which_universities_offer_a_course(rag_module):
    result = rag_module.answer_question("Which universities offer a Master of AI?")
    assert {c.metadata["course_key"] for c in result.evidence} == {"ai"}
    assert "No Master of Artificial Intelligence record" in result.answer


def test_every_structured_record_has_a_known_course(rag_module):
    import config

    metadatas = rag_module.get_collection().get(where={"doc_type": "structured_course"}, include=["metadatas"])["metadatas"]
    assert len(metadatas) == 33
    assert {m["course_key"] for m in metadatas} <= set(config.COURSE_REGISTRY)


def test_every_new_course_has_a_snapshot_pdf():
    import csv
    from pathlib import Path

    import config
    import ingest_clean_v2 as ingest

    folder_for = {entry["name"]: key for key, entry in config.UNIVERSITY_REGISTRY.items()}
    with (config.STRUCTURED_DIR / "university_courses.csv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = row["course_key"]
            name = "course.pdf" if key == "data_science" else f"course_{key}.pdf"
            path = Path(config.RAW_DIR) / folder_for[row["university"]] / name
            assert path.exists(), path
            assert ingest.infer_metadata(path)["course_key"] == key


def test_fee_answer_does_not_quote_verification_notes(rag_module):
    result = rag_module.answer_question("What is the fee for the Master of IT at RMIT?")
    assert "38,400" not in result.answer
    assert "Course page snapshot says" in result.answer
