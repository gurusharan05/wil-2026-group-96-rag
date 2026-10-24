"""Core retrieval and answer generation for the International Student RAG.

Pipeline (see CHANGES.md for why it is built this way):

    question
      -> plan_query()      which universities / topics / facts are asked for
      -> retrieve()        fetch ONLY the evidence for that plan
                             * fact questions (fee, duration, campus, ...):
                               direct metadata lookup of the structured course
                               record for the named university (no embedding,
                               no LLM) + the source-page chunk that states it
                             * other questions: vector search restricted to the
                               named university and topic
      -> generate_answer() deterministic answer when the evidence is structured,
                           otherwise a grounded LLM answer (thinking disabled)

`answer_question()` runs the whole pipeline and returns the answer together
with the exact evidence it used, so the UI never shows unrelated sources.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Iterator, Optional

import chromadb
import ollama

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    COURSE_NAMES,
    COURSE_REGISTRY,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    GENERAL_ENTITY,
    INDEX_VERSION,
    LLM_MODEL,
    LLM_NUM_CTX,
    LLM_NUM_PREDICT,
    MAX_DISTANCE,
    OLLAMA_KEEP_ALIVE,
    UNIVERSITY_REGISTRY,
)


# =========================================================
# Data classes
# =========================================================

@dataclass
class RetrievedChunk:
    """A retrieved knowledge-base chunk."""
    text: str
    distance: float
    metadata: dict


@dataclass
class QueryPlan:
    """What the question is asking for."""
    universities: list = field(default_factory=list)
    topics: list = field(default_factory=list)
    facts: list = field(default_factory=list)
    courses: list = field(default_factory=list)   # course keys named in the question
    is_comparison: bool = False
    university_explicit: bool = False


@dataclass
class RAGResult:
    """Answer + the evidence actually used to produce it."""
    answer: str
    evidence: list
    route: str
    plan: QueryPlan
    timings: dict = field(default_factory=dict)
    stream: Optional[Iterator[str]] = None


class IndexOutdatedError(RuntimeError):
    """The ChromaDB index was built by an older ingestion script."""


# =========================================================
# ChromaDB access (cached — opening the DB on every question was slow)
# =========================================================

@lru_cache(maxsize=1)
def _client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


@lru_cache(maxsize=1)
def _cached_collection():
    try:
        collection = _client().get_collection(name=COLLECTION_NAME)
    except Exception as error:
        raise IndexOutdatedError(
            f"The knowledge base collection '{COLLECTION_NAME}' was not found in "
            f"{CHROMA_DIR} ({error}). Build it with:\n\n"
            "    PYTHONPATH=. python scripts/ingest_clean_v2.py"
        ) from error
    version = (collection.metadata or {}).get("index_version")
    if version != INDEX_VERSION:
        raise IndexOutdatedError(
            "The knowledge base in data/chroma_db was built by an older version of "
            "the ingestion script (it labels ACU, Deakin, La Trobe and Swinburne as "
            "'Australia'). Rebuild it with:\n\n"
            "    PYTHONPATH=. python scripts/ingest_clean_v2.py"
        )
    return collection


def get_collection():
    """Return the (cached) Chroma collection.

    Document embeddings were generated with Ollama during ingestion, so query
    embeddings are also generated manually with the same model.
    """
    return _cached_collection()


def reset_cache() -> None:
    """Forget cached DB handles and embeddings (call after re-ingesting)."""
    _cached_collection.cache_clear()
    _client.cache_clear()
    _embed_cached.cache_clear()


def _with_retry(fn):
    """Run a DB call; if the collection was rebuilt underneath us, reconnect once."""
    try:
        return fn(get_collection())
    except IndexOutdatedError:
        raise
    except Exception:
        reset_cache()
        return fn(get_collection())


# =========================================================
# Embeddings (cached per query string)
# =========================================================

@lru_cache(maxsize=512)
def _embed_cached(texts: tuple) -> tuple:
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=list(texts),
        keep_alive=OLLAMA_KEEP_ALIVE,
    )
    return tuple(tuple(vector) for vector in response["embeddings"])


def embed_queries(texts: list) -> list:
    if not texts:
        return []
    return [list(vector) for vector in _embed_cached(tuple(texts))]


def warm_up() -> dict:
    """Load both Ollama models into memory so the first question is not slow."""
    status = {}
    try:
        ollama.embed(model=EMBEDDING_MODEL, input=["warm up"], keep_alive=OLLAMA_KEEP_ALIVE)
        status["embedding_model"] = "ready"
    except Exception as error:  # pragma: no cover - depends on local Ollama
        status["embedding_model"] = f"unavailable: {error}"
    try:
        # An empty message list loads the model without generating anything.
        ollama.chat(model=LLM_MODEL, messages=[], keep_alive=OLLAMA_KEEP_ALIVE)
        status["llm"] = "ready"
    except Exception as error:  # pragma: no cover
        status["llm"] = f"unavailable: {error}"
    return status


# =========================================================
# Question understanding
# =========================================================

def _compile_alias_patterns():
    patterns = []
    for entry in UNIVERSITY_REGISTRY.values():
        aliases = sorted(set(a.casefold() for a in entry["aliases"] + [entry["name"]]), key=len, reverse=True)
        regex = re.compile(
            r"(?<![a-z0-9])(?:" + "|".join(re.escape(a) for a in aliases) + r")(?![a-z0-9])"
        )
        patterns.append((entry["name"], regex))
    return patterns


_ALIAS_PATTERNS = _compile_alias_patterns()


def get_mentioned_universities(question: str) -> list:
    """Universities named in the question, in the order they are mentioned.

    Uses the fixed registry in config.py rather than scanning every record in
    ChromaDB on every question (which was slow and depended on the stored
    labels being correct). Matching is on word boundaries, so "acu" does not
    match inside another word and plain "Melbourne" (the city) does not match
    The University of Melbourne.
    """
    q = question.casefold()
    found = []
    for name, regex in _ALIAS_PATTERNS:
        match = regex.search(q)
        if match:
            found.append((match.start(), name))
    return [name for _, name in sorted(found)]


def detect_university(question: str) -> Optional[str]:
    """Return the first explicitly mentioned university, if any."""
    found = get_mentioned_universities(question)
    return found[0] if found else None


def _compile_course_patterns():
    patterns = []
    for key, entry in COURSE_REGISTRY.items():
        aliases = sorted(set(a.casefold() for a in entry["aliases"] + [entry["name"]]), key=len, reverse=True)
        insensitive = re.compile(
            r"(?<![a-z0-9])(?:" + "|".join(re.escape(a) for a in aliases) + r")(?![a-z0-9])"
        )
        sensitive = None
        if entry.get("case_sensitive"):
            sensitive = re.compile(
                r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(a) for a in entry["case_sensitive"]) + r")(?![A-Za-z0-9])"
            )
        patterns.append((key, insensitive, sensitive))
    return patterns


_COURSE_PATTERNS = _compile_course_patterns()
# Words that look like a course alias but are part of a university name.
_COURSE_FALSE_FRIENDS = re.compile(r"royal melbourne institute of technology")


def detect_courses(question: str) -> list:
    """Course keys named in the question, in the order they are mentioned.

    "IT" and "AI" only count when typed in capitals (or as part of a longer
    phrase such as "master of it"), so "is it online?" is not an IT question.
    """
    q = _COURSE_FALSE_FRIENDS.sub(" ", question.casefold())
    found = []
    for key, insensitive, sensitive in _COURSE_PATTERNS:
        positions = []
        match = insensitive.search(q)
        if match:
            positions.append(match.start())
        if sensitive is not None:
            match = sensitive.search(question)
            if match:
                positions.append(match.start())
        if key == "ai":
            # "ai" in lower case is only accepted inside a course phrase.
            if not FACT_SAFE_AI.search(q) and not (sensitive and sensitive.search(question)) \
                    and not re.search(r"artificial intelligence", q):
                positions = []
        if positions:
            found.append((min(positions), key))
    return [key for _, key in sorted(found)]


def _rx(*words: str):
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(words) + r")(?![a-z0-9])")


# Word-boundary patterns. The old substring checks matched "rent" inside
# "current"/"different", "work" inside "coursework" and "fee" inside "coffee",
# which sent simple fee questions down the accommodation / work-rights / LLM
# paths and pulled in unrelated sources.
FACT_PATTERNS = {
    "fee": _rx(r"fees?", r"tuition", r"course cost", r"cost of (?:the )?(?:course|degree|program(?:me)?|master'?s?)"),
    "duration": _rx(r"durations?", r"how long", r"how many years", r"length of (?:the )?(?:course|degree|program(?:me)?)",
                    r"(?:full|part)[- ]time duration", r"years? (?:does|will) it take", r"accelerated"),
    "location": _rx(r"campus(?:es)?", r"location", r"located", r"where is (?:the )?(?:course|program|degree|it)",
                    r"where (?:is|are) .* (?:taught|offered|delivered)", r"which campus", r"offered", r"online"),
    "intake": _rx(r"intakes?", r"start dates?", r"start months?", r"when (?:can|do) (?:i|you|students) start",
                  r"commence(?:ment)?", r"starting"),
    "course_code": _rx(r"course code", r"program code", r"programme code"),
    "course_name": _rx(r"name of (?:the |its |their )?(?:[a-z']+ ){0,6}(?:course|degree|program(?:me)?)", r"course name", r"course title", r"what is (?:the )?(?:course|degree) called"),
    "cricos": _rx(r"cricos"),
    "entry": _rx(r"entry requirements?", r"admission(?:s)?(?: requirements?)?", r"eligib(?:le|ility)",
                 r"prerequisites?", r"wam", r"gpa", r"minimum average", r"average mark", r"requirements?"),
    "english": _rx(r"ielts", r"english", r"pte", r"toefl"),
}

# "Does Swinburne offer a Master of AI?" -> answer with the course overview.
AVAILABILITY_PATTERN = _rx(r"offers?", r"offered", r"have", r"has", r"available", r"provides?", r"runs?",
                           r"is there", r"are there", r"exists?")

FACT_SAFE_AI = _rx(r"master'?s? (?:of |in )?ai", r"ai (?:course|degree|masters?|program(?:me)?)",
                   r"ai strategy")

# Weak price words only count as a course-fee question when the question is
# not about housing or jobs ("how much is rent" is not a tuition question).
WEAK_FEE_PATTERN = _rx(r"costs?", r"price", r"how much", r"expensive", r"cheap(?:er|est)?", r"afford(?:able)?")

COURSE_PATTERN = _rx(r"courses?", r"degrees?", r"master'?s?", r"data science", r"program(?:me)?s?",
                     r"subjects?", r"curriculum", r"specialisations?", r"specializations?", r"study plan")
ACCOMMODATION_PATTERN = _rx(r"accommodation", r"housing", r"rent(?:al|als|ing)?", r"residences?", r"residential",
                            r"colleges?", r"homestay", r"dorm(?:itory|s)?", r"apartments?", r"share ?hou?ses?",
                            r"where (?:to|can i|should i) live", r"place to live", r"on[- ]campus living")
WORK_PATTERN = _rx(r"work(?:ing|s)?", r"jobs?", r"employment", r"employers?", r"hours", r"fortnight",
                   r"wages?", r"salary", r"minimum pay", r"pay rates?", r"payslips?", r"visa",
                   r"workplace", r"fair work", r"super(?:annuation)?", r"work rights", r"part[- ]time (?:work|job)s?")
INTERNATIONAL_PATTERN = _rx(r"support services?", r"student support", r"orientation", r"arrival", r"welfare",
                            r"oshc", r"health cover", r"international student (?:services|support|information)")

COMPARISON_PATTERN = _rx(r"compare", r"comparison", r"comparing", r"versus", r"vs\.?", r"difference",
                         r"choose between", r"choosing between", r"which (?:one|is|uni|university|universities)",
                         r"cheapest", r"cheaper", r"most expensive", r"lowest", r"highest", r"best")
ALL_UNIVERSITIES_PATTERN = _rx(r"universities", r"all (?:the )?unis?", r"each uni(?:versity)?",
                               r"every uni(?:versity)?", r"these unis?", r"which uni(?:versity)?",
                               r"unis in melbourne", r"university options", r"university choice",
                               r"choos(?:e|ing) a university", r"course comparison")


def detect_facts(question: str) -> list:
    q = question.casefold()
    facts = [name for name, pattern in FACT_PATTERNS.items() if pattern.search(q)]
    housing_or_work = ACCOMMODATION_PATTERN.search(q) or WORK_PATTERN.search(q)
    if "fee" not in facts and WEAK_FEE_PATTERN.search(q) and (not housing_or_work or COURSE_PATTERN.search(q)):
        facts.insert(0, "fee")
    # "online" / "offered" / "requirements" are only course facts in a course context.
    if housing_or_work and not COURSE_PATTERN.search(q):
        # Only unambiguous course facts survive in a housing / work question.
        facts = [f for f in facts if f in {"fee", "course_code", "course_name", "cricos"}]
    return facts


def detect_topics(question: str, facts: Optional[list] = None) -> list:
    q = question.casefold()
    facts = detect_facts(question) if facts is None else facts
    topics = []
    if facts or COURSE_PATTERN.search(q) or detect_courses(question):
        topics.append("course")
    if ACCOMMODATION_PATTERN.search(q):
        topics.append("accommodation")
    if WORK_PATTERN.search(q):
        topics.append("work_rights")
    if INTERNATIONAL_PATTERN.search(q):
        topics.append("international_students")
    return topics or ["general"]


def plan_query(question: str, university: str = "All", category: str = "All") -> QueryPlan:
    """Decide which universities, topics and facts a question needs."""
    q = question.casefold()
    facts = detect_facts(question)
    topics = detect_topics(question, facts)
    universities = get_mentioned_universities(question)
    explicit = bool(universities)

    if university and university != "All":
        universities = [] if university == GENERAL_ENTITY else [university]
        explicit = bool(universities)

    # "...work while the course is in session?" is a work-rights question,
    # not a course question: a bare mention of "course" with no university and
    # no course fact must not pull every university's course pages.
    if "course" in topics and len(topics) > 1 and not facts and not universities:
        topics = [t for t in topics if t != "course"]

    if category and category != "All":
        topics = [category]

    wants_all = bool(ALL_UNIVERSITIES_PATTERN.search(q))
    compare_words = bool(COMPARISON_PATTERN.search(q))

    if not universities and "course" in topics and (wants_all or compare_words or facts):
        # "What are the fees?" / "Which university is cheapest?" without naming a
        # university: answer for every university from the structured records.
        universities = [entry["name"] for entry in UNIVERSITY_REGISTRY.values()]

    courses = detect_courses(question)
    if courses and not facts and AVAILABILITY_PATTERN.search(q) and "course" in topics:
        facts = ["location", "duration", "fee"]
        if not universities:
            universities = [entry["name"] for entry in UNIVERSITY_REGISTRY.values()]
    is_comparison = "course" in topics and len(universities) >= 2
    return QueryPlan(
        universities=universities,
        topics=topics,
        facts=facts,
        courses=courses,
        is_comparison=is_comparison,
        university_explicit=explicit,
    )


# =========================================================
# Low-level retrieval helpers
# =========================================================

def _where(*conditions: dict) -> Optional[dict]:
    conditions = [c for c in conditions if c]
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": list(conditions)}


def _university_condition(universities: list) -> Optional[dict]:
    if not universities:
        return None
    if len(universities) == 1:
        return {"university": universities[0]}
    return {"university": {"$in": list(universities)}}


def _course_condition(courses: list) -> Optional[dict]:
    if not courses:
        return None
    if len(courses) == 1:
        return {"course_key": courses[0]}
    return {"course_key": {"$in": list(courses)}}


def _course_order(key: str) -> int:
    keys = list(COURSE_REGISTRY)
    return keys.index(key) if key in keys else len(keys)


def _get(where: Optional[dict], tag: str) -> list:
    """Metadata lookup (no embedding)."""
    def run(collection):
        return collection.get(where=where, include=["documents", "metadatas"])

    result = _with_retry(run)
    chunks = []
    for text, metadata in zip(result.get("documents") or [], result.get("metadatas") or []):
        metadata = dict(metadata or {})
        metadata["_retrieval"] = tag
        chunks.append(RetrievedChunk(text=text, distance=0.0, metadata=metadata))
    return chunks


def _query(embedding: list, where: Optional[dict], n_results: int, tag: str) -> list:
    """Vector search."""
    def run(collection):
        return collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    results = _with_retry(run)
    chunks = []
    for text, distance, metadata in zip(
        (results.get("documents") or [[]])[0],
        (results.get("distances") or [[]])[0],
        (results.get("metadatas") or [[]])[0],
    ):
        metadata = dict(metadata or {})
        metadata["_retrieval"] = tag
        chunks.append(RetrievedChunk(text=text, distance=float(distance), metadata=metadata))
    return chunks


def _is_structured(chunk: RetrievedChunk) -> bool:
    return chunk.metadata.get("doc_type") == "structured_course" or chunk.metadata.get("source_type") == "csv"


def _registry_order(name: str) -> int:
    names = [entry["name"] for entry in UNIVERSITY_REGISTRY.values()]
    return names.index(name) if name in names else len(names)


# =========================================================
# Fact extraction
# =========================================================

STRUCTURED_FIELDS = {
    "fee": ["International fee", "Fee year", "Fee basis"],
    "duration": ["Full-time duration", "Part-time duration", "Advanced standing duration"],
    "location": ["Campus", "Study mode"],
    "intake": ["Published intake"],
    "course_code": ["Course code"],
    "course_name": ["Course"],
    "cricos": ["CRICOS code"],
    "entry": ["Entry requirements"],
    "english": ["English requirements"],
}

FACT_LABELS = {
    "fee": "International fee",
    "duration": "Duration",
    "location": "Campus / mode",
    "intake": "Intake",
    "course_code": "Course code",
    "course_name": "Course name",
    "cricos": "CRICOS code",
    "entry": "Entry requirements",
    "english": "English requirements",
}

# Words that identify a sentence in a source page as stating a given fact.
FACT_LINE_WORDS = {
    "fee": re.compile(r"\b(fee|fees|tuition)\b", re.I),
    "duration": re.compile(r"\b(duration|years? full-time|full-time)\b", re.I),
    "location": re.compile(r"\b(campus|location|parkville|clayton|burwood|bundoora|hawthorn|melbourne city|online)\b", re.I),
    "intake": re.compile(r"\b(intakes?|start dates?|semester \d|february|march|july|august|november)\b", re.I),
    "course_code": re.compile(r"\b(course code|program code)\b", re.I),
    "course_name": re.compile(r"\bcourse:", re.I),
    "cricos": re.compile(r"\bcricos\b", re.I),
    "entry": re.compile(r"\b(wam|gpa|average|bachelor|entry level|entry requirements?)\b", re.I),
    "english": re.compile(r"\b(ielts|english)\b", re.I),
}

MONEY_PATTERN = re.compile(r"(?:AUD|A\$|\$)\s?\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)?|(?:AUD|A\$|\$)\s?\d{4,}")


def parse_structured_record(text: str) -> dict:
    """'Key: value' lines of a structured course record -> dict."""
    record = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            record[key.strip()] = value.strip()
    return record


def _logical_lines(text: str) -> list:
    """Re-join PDF lines that were wrapped mid-sentence."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # A PDF line that wrapped mid-sentence is long (~90+ chars) and has no
        # closing punctuation; anything else starts a new logical line.
        starts_new = (
            not out
            or out[-1].endswith((".", ":", ";", "!", "?"))
            or len(out[-1]) < 70
            or out[-1].startswith(_SKIP_PREFIXES)
            or re.match(r"^[A-Z][A-Za-z /()-]{1,40}:", line)
        )
        if starts_new:
            out.append(line)
        else:
            out[-1] = f"{out[-1]} {line}"
    return out


_SKIP_PREFIXES = ("Entity:", "Snapshot date:", "Source URL:", "Important:", "RAG use", "Use this source")


def fact_sentences(chunk: RetrievedChunk, fact: str, limit: int = 3) -> list:
    """Sentences in a source-page chunk that state the requested fact."""
    pattern = FACT_LINE_WORDS.get(fact)
    if pattern is None:
        return []
    found = []
    for line in _logical_lines(chunk.text):
        if line.casefold().startswith("verification notes"):
            break  # notes about what to check are not facts to quote
        if line.startswith(_SKIP_PREFIXES) or len(line) < 12:
            continue
        if not pattern.search(line):
            continue
        if fact == "fee" and not MONEY_PATTERN.search(line):
            continue  # a fee sentence must contain an amount
        found.append(line)
        if len(found) >= limit:
            break
    return found


# =========================================================
# Retrieval
# =========================================================

def _structured_records(universities: list, courses: Optional[list] = None) -> list:
    chunks = _get(
        _where({"doc_type": "structured_course"}, _university_condition(universities), _course_condition(courses or [])),
        "direct",
    )
    for chunk in chunks:
        chunk.metadata["_requested_university"] = chunk.metadata.get("university")
    chunks.sort(key=lambda c: (_course_order(c.metadata.get("course_key", "")),
                               _registry_order(c.metadata.get("university", ""))))
    return chunks


def _supporting_page_chunks(university: str, facts: list, course_key: str = "") -> list:
    """Source-page chunks of this university (and course) that state the requested facts."""
    pages = _get(
        _where({"university": university}, {"category": "course"}, {"doc_type": "document"},
               {"course_key": course_key} if course_key else None),
        "direct",
    )
    supporting = []
    for chunk in pages:
        hits = [fact for fact in facts if fact_sentences(chunk, fact)]
        if hits:
            chunk.metadata["_supports"] = ",".join(hits)
            supporting.append((len(hits), chunk))
    supporting.sort(key=lambda item: -item[0])
    return [chunk for _, chunk in supporting[:1]]


def _vector_search(question: str, where: Optional[dict], n_results: int) -> list:
    [embedding] = embed_queries([question])
    return _query(embedding, where, n_results, "vector" if where is None else "vector_filtered")


def _is_low_value(chunk: RetrievedChunk) -> bool:
    section = str(chunk.metadata.get("section", "")).strip().lower()
    return section == "rag use" or len(chunk.text.strip()) < 120


def retrieve(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    university: str = "All",
    category: str = "All",
    plan: Optional[QueryPlan] = None,
) -> list:
    """Return only the evidence needed for this question.

    Every request is restricted to the universities the question names, so a
    question about RMIT can no longer pull chunks from another university.
    """
    if top_k <= 0:
        return []
    plan = plan or plan_query(question, university, category)
    selected = []
    seen = set()

    def add(chunks: Iterable[RetrievedChunk], topic: str) -> None:
        for chunk in chunks:
            key = (chunk.text, chunk.metadata.get("source_file"), chunk.metadata.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            chunk.metadata["_topic"] = topic
            selected.append(chunk)

    for topic in plan.topics:
        if topic == "course":
            if plan.universities and (plan.facts or plan.is_comparison or not plan.courses):
                records = _structured_records(plan.universities, plan.courses)
                if plan.is_comparison or len(records) > 1:
                    # Put the record that matches the question best first
                    # (e.g. "...at Clayton campus?" -> Monash first).
                    words = {w for w in re.findall(r"[a-z]{4,}", question.casefold())}
                    records.sort(key=lambda c: -sum(1 for w in words if w in c.text.casefold()))
                    add(records, topic)
                else:
                    for record in records:
                        add([record], topic)
                        add(_supporting_page_chunks(record.metadata["university"], plan.facts or ["fee"],
                                                    record.metadata.get("course_key", "")), topic)
                missing = [u for u in plan.universities
                           if u not in {r.metadata.get("university") for r in records}]
                if missing and not plan.is_comparison:
                    # No structured record for this university: fall back to a
                    # semantic search restricted to that university's course pages.
                    where = _where(_university_condition(missing), {"category": "course"},
                                   _course_condition(plan.courses))
                    add([c for c in _vector_search(question, where, top_k) if not _is_low_value(c)], topic)
            elif plan.universities:
                where = _where(_university_condition(plan.universities), {"category": "course"},
                               _course_condition(plan.courses))
                add([c for c in _vector_search(question, where, top_k) if not _is_low_value(c)], topic)
                add(_structured_records(plan.universities[:1], plan.courses), topic)
            else:
                add(_vector_search(question, _where({"category": "course"}, _course_condition(plan.courses)), top_k),
                    topic)

        elif topic in {"accommodation", "international_students"}:
            if plan.universities:
                add(_get(_where(_university_condition(plan.universities), {"category": topic}), "direct"), topic)
            else:
                add(_vector_search(question, {"category": topic}, 3), topic)

        elif topic == "work_rights":
            where = _where({"university": GENERAL_ENTITY}, {"category": {"$in": ["work_rights", "general"]}})
            add(_vector_search(question, where, 2), topic)

        else:  # general
            where = _university_condition(plan.universities)
            add([c for c in _vector_search(question, where, top_k) if not _is_low_value(c)], topic)

    if plan.is_comparison or sum(1 for c in selected if _is_structured(c)) > 1:
        return selected
    # Facts route returns at most (record + supporting page) per university.
    limit = max(top_k, 2 * len(plan.universities)) if plan.facts else top_k
    return selected[:limit]


# =========================================================
# Deterministic answers
# =========================================================

def _clean_fee(record: dict) -> str:
    fee = record.get("International fee", "")
    if not fee or fee.lower().startswith("not "):
        return ""
    return fee


def _fact_value(record: dict, fact: str) -> str:
    if fact == "fee":
        fee = _clean_fee(record)
        if not fee:
            return ""
        year = record.get("Fee year", "")
        year_text = f" ({year})" if year and year.lower() not in {"not stated", "not specified"} else ""
        return f"{fee}{year_text}"
    if fact == "duration":
        parts = [f"Full-time: {record.get('Full-time duration', 'Not stated')}"]
        part_time = record.get("Part-time duration", "")
        if part_time and part_time.lower() != "not stated":
            parts.append(f"Part-time: {part_time}")
        return " · ".join(parts)
    if fact == "location":
        campus = record.get("Campus", "Not stated")
        mode = record.get("Study mode", "")
        if not mode or mode.lower() == "not stated" or mode.casefold() == campus.casefold():
            return campus
        return f"{campus} · {mode}"
    key = STRUCTURED_FIELDS[fact][0]
    value = record.get(key, "")
    return "" if value.lower() == "not stated" else value


def _single_university_fact_answer(record_chunk: RetrievedChunk, page_chunks: list, facts: list) -> str:
    record = parse_structured_record(record_chunk.text)
    university = record.get("University", record_chunk.metadata.get("university", ""))
    course = record.get("Course", "Master of Data Science")
    lines = [f"**{university} — {course}**", ""]

    for fact in facts:
        value = _fact_value(record, fact)
        page_lines = []
        for page in page_chunks:
            page_lines.extend(fact_sentences(page, fact, limit=2))

        if value:
            lines.append(f"- **{FACT_LABELS[fact]}:** {value}")
            if fact == "fee":
                basis = record.get("Fee basis", "")
                if basis and basis.lower() != "not stated":
                    lines.append(f"- **Fee basis:** {basis}")
            if fact == "duration":
                adv = record.get("Advanced standing duration", "")
                if adv and adv.lower() != "not stated":
                    lines.append(f"- **Advanced standing:** {adv}")
        elif page_lines:
            lines.append(f"- **{FACT_LABELS[fact]}:** {page_lines[0]}")
        else:
            lines.append(
                f"- **{FACT_LABELS[fact]}:** The knowledge base does not specify this for {university}."
            )

        if fact == "fee" and value and page_lines:
            lines.append("- **Course page snapshot says:** " + " ".join(f"“{line}”" for line in page_lines))

    if _needs_verification(record):
        lines.append("")
        lines.append("_This record was collected from official pages on 2026-10-19 and is awaiting team "
                     "verification._")
    url = record.get("Official course URL", "")
    if "fee" in facts:
        lines.append("")
        lines.append("_Fees are year-specific and change annually — confirm on the official course page"
                     + (f": {url}_" if url and url.lower() != "not stated" else "._"))
    return "\n".join(lines).strip()


def _needs_verification(record: dict) -> bool:
    return bool(record.get("Verification note"))


def _records_table(records: list, facts: list, first_col: str) -> list:
    """Markdown table: one row per structured record."""
    show_course = first_col == "University" and len({parse_structured_record(c.text).get("Course") for c in records}) > 1
    head = [first_col] + (["Course"] if show_course else []) + [FACT_LABELS[f] for f in facts]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for chunk in records:
        record = parse_structured_record(chunk.text)
        mark = " †" if _needs_verification(record) else ""
        label = record.get("University") if first_col == "University" else record.get("Course")
        cells = [f"{label}{mark}"]
        if show_course:
            cells.append(record.get("Course", ""))
        cells += [(_fact_value(record, f) or "Not specified").replace("|", "/") for f in facts]
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def _fee_basis_notes(records: list, label_for) -> list:
    notes = ["", "**Fee bases differ** (annual, per EFTSL, per credit points, or total course), so these "
             "figures are not directly comparable:"]
    for chunk in records:
        record = parse_structured_record(chunk.text)
        notes.append(f"- {label_for(record)}: {record.get('Fee basis', 'Not stated')}")
    return notes


def _verification_footnote(records: list) -> list:
    if any(_needs_verification(parse_structured_record(c.text)) for c in records):
        return ["", "_† Collected from official pages on 2026-10-19 and awaiting team verification — "
                "confirm on the official course page before relying on it._"]
    return []


def _missing_note(plan: QueryPlan, records: list) -> list:
    """Say which requested university/course pairs have no record."""
    if not plan.courses:
        return []
    have = {(c.metadata.get("university"), c.metadata.get("course_key")) for c in records}
    lines = []
    for key in plan.courses:
        missing = [u for u in plan.universities if (u, key) not in have]
        if missing:
            lines.append(f"_No {COURSE_NAMES.get(key, key)} record in the knowledge base for: "
                         + ", ".join(missing) + "._")
    return ([""] + lines) if lines else []


def _comparison_answer(records: list, facts: list, plan: Optional[QueryPlan] = None) -> str:
    """Several universities: one table per course."""
    facts = facts or ["fee", "duration", "location", "intake"]
    out = []
    groups = {}
    for chunk in records:
        groups.setdefault(chunk.metadata.get("course_key", ""), []).append(chunk)
    for key in sorted(groups, key=_course_order):
        group = groups[key]
        if len(groups) > 1:
            out += [f"#### {COURSE_NAMES.get(key, key)}", ""]
        out += _records_table(group, facts, "University")
        if "fee" in facts:
            out += _fee_basis_notes(group, lambda r: r.get("University"))
        out.append("")
    out += _verification_footnote(records)
    if plan is not None:
        out += _missing_note(plan, records)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _course_table_answer(records: list, facts: list, plan: Optional[QueryPlan] = None) -> str:
    """One university, several courses."""
    facts = facts or ["fee", "duration", "location", "intake"]
    university = parse_structured_record(records[0].text).get("University")
    out = [f"**{university} — {len(records)} courses in the knowledge base**", ""]
    out += _records_table(records, facts, "Course")
    if "fee" in facts:
        out += _fee_basis_notes(records, lambda r: r.get("Course"))
    out += _verification_footnote(records)
    if plan is not None:
        out += _missing_note(plan, records)
    out += ["", "_Ask about one course (e.g. “RMIT Master of IT fee”) for full details and sources._"]
    return "\n".join(out).strip()


def _document_dump(chunks: list, title_for, topic: str = "") -> str:
    """Bullet out the substantive lines of accommodation / work-rights pages.

    Some government pages cover both work and housing, so lines that clearly
    belong to the *other* topic are left out.
    """
    off_topic = {"work_rights": ACCOMMODATION_PATTERN, "accommodation": WORK_PATTERN}.get(topic)
    on_topic = {"work_rights": WORK_PATTERN, "accommodation": ACCOMMODATION_PATTERN}.get(topic)
    groups = {}
    for chunk in chunks:
        groups.setdefault(title_for(chunk), []).append(chunk)
    out = []
    for title, group in groups.items():
        out.append(f"### {title}")
        seen = set()
        for chunk in group:
            for i, line in enumerate(_logical_lines(chunk.text)):
                if i == 0 or line.startswith(_SKIP_PREFIXES) or line in seen:
                    continue
                if len(line) < 45 and not line.endswith("."):
                    continue  # section headings
                lowered = line.casefold()
                if off_topic and off_topic.search(lowered) and not on_topic.search(lowered):
                    continue
                seen.add(line)
                out.append(f"- {line}")
        out.append("")
    return "\n".join(out).strip()


def _deterministic_answer(plan: QueryPlan, chunks: list) -> Optional[str]:
    """Answer without the LLM when the evidence is structured. None = needs LLM."""
    sections = []
    for topic in plan.topics:
        topic_chunks = [c for c in chunks if c.metadata.get("_topic", topic) == topic]
        if topic == "course":
            records = [c for c in topic_chunks if _is_structured(c)]
            if plan.courses and plan.universities and not records and (plan.facts or plan.is_comparison):
                # Asked for a course a university does not have in the knowledge base.
                sections.append("\n".join(_missing_note(plan, records)).strip())
                continue
            table = len(records) > 1 and not plan.courses
            if not (plan.facts or plan.is_comparison or table) or not records:
                return None
            if plan.is_comparison:
                sections.append(_comparison_answer(records, plan.facts, plan))
            elif len(records) > 1:
                sections.append(_course_table_answer(records, plan.facts, plan))
            else:
                if plan.courses and plan.universities:
                    note = _missing_note(plan, records)
                    if note:
                        sections.append("\n".join(note).strip())
                for record in records:
                    university = record.metadata.get("university")
                    pages = [c for c in topic_chunks if not _is_structured(c) and c.metadata.get("university") == university]
                    sections.append(_single_university_fact_answer(record, pages, plan.facts))
        elif topic in {"accommodation", "international_students"}:
            if not topic_chunks:
                return None
            label = "accommodation" if topic == "accommodation" else "international students"
            sections.append(_document_dump(
                topic_chunks, lambda c: f"{c.metadata.get('university')} — {label}", topic))
        elif topic == "work_rights":
            if not topic_chunks:
                return None
            sections.append(_document_dump(
                topic_chunks, lambda c: c.metadata.get("authority") or "Work rights", topic))
        else:
            return None
    return "\n\n".join(s for s in sections if s).strip() or None


def _fast_fact_answer(question: str, chunks: list) -> Optional[str]:
    """Backwards-compatible wrapper used by older scripts."""
    return _deterministic_answer(plan_query(question), chunks)


# =========================================================
# LLM answers
# =========================================================

def build_context(chunks: list) -> str:
    """Convert RetrievedChunk objects into the grounded context supplied to the LLM."""
    if not chunks:
        return "No relevant information was retrieved."
    sections = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.metadata or {}
        header = (
            f"[Source {i}] University: {metadata.get('university', 'Unknown')} | "
            f"Category: {metadata.get('category', 'Unknown')} | "
            f"Section: {metadata.get('section', 'Unknown')}"
        )
        sections.append(f"{header}\n{chunk.text}")
    return "\n\n".join(sections)


SYSTEM_PROMPT = (
    "You are a precise, evidence-grounded assistant for international students. "
    "Answer only from the supplied sources, cite them as [Source N], and never invent facts."
)


def _build_prompt(question: str, chunks: list, plan: QueryPlan) -> str:
    scope = ""
    if plan.universities:
        scope = "Only discuss: " + ", ".join(plan.universities) + ".\n"
    if plan.courses:
        scope += "Only discuss these courses: " + ", ".join(COURSE_NAMES[k] for k in plan.courses) + ".\n"
    return f"""USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{build_context(chunks)}

RULES:
1. Answer the question directly and concisely (aim for under 150 words).
2. Use only facts in the evidence; keep numbers exactly as written.
3. Answer every part of a multi-part question.
4. If something is not in the evidence, say "The retrieved evidence does not specify this."
5. Cite claims with the source labels shown, e.g. [Source 1]. Do not invent source numbers.
6. Do not recommend one university as objectively best.
{scope}"""


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _chat(messages: list, stream: bool):
    options = {
        "temperature": 0.0,
        "num_predict": LLM_NUM_PREDICT,
        "num_ctx": LLM_NUM_CTX,
    }
    try:
        # `think` is a top-level argument of ollama.chat(). It used to be
        # passed inside `options`, where Ollama silently ignores it, so qwen3
        # generated a long hidden reasoning trace before every answer.
        return ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            think=False,
            stream=stream,
            options=options,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
    except (TypeError, ollama.ResponseError):
        # Older Ollama server/client without the `think` flag: use qwen3's
        # soft switch instead.
        patched = [dict(m) for m in messages]
        patched[-1]["content"] += "\n/no_think"
        return ollama.chat(
            model=LLM_MODEL,
            messages=patched,
            stream=stream,
            options=options,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )


def _messages(question: str, chunks: list, plan: QueryPlan) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(question, chunks, plan)},
    ]


def _message_content(part) -> str:
    message = part["message"] if isinstance(part, dict) else part.message
    return (message["content"] if isinstance(message, dict) else message.content) or ""


def _stream_tokens(question: str, chunks: list, plan: QueryPlan) -> Iterator[str]:
    """Yield answer text as it is generated, hiding any <think> block."""
    buffer = ""
    in_think = False
    for part in _chat(_messages(question, chunks, plan), stream=True):
        buffer += _message_content(part)
        while buffer:
            if in_think:
                end = buffer.find("</think>")
                if end == -1:
                    buffer = buffer[-8:]
                    break
                buffer = buffer[end + len("</think>"):]
                in_think = False
            else:
                start = buffer.find("<think>")
                if start == -1:
                    # keep a few chars in case a tag is split across parts
                    safe = len(buffer) - 7 if "<" in buffer[-7:] else len(buffer)
                    if safe > 0:
                        yield buffer[:safe]
                        buffer = buffer[safe:]
                    break
                if start:
                    yield buffer[:start]
                buffer = buffer[start + len("<think>"):]
                in_think = True
    if buffer and not in_think:
        yield buffer


NO_EVIDENCE = "I could not find relevant information in the knowledge base for this question."


def _below_threshold(chunks: list) -> bool:
    """True when an unrestricted semantic search found nothing close enough."""
    unrestricted = [c for c in chunks if c.metadata.get("_retrieval") == "vector"]
    if not unrestricted or len(unrestricted) != len(chunks):
        return False
    return min(c.distance for c in unrestricted) > MAX_DISTANCE


def generate_answer(question: str, chunks: list, plan: Optional[QueryPlan] = None) -> str:
    """Generate a grounded answer from the retrieved evidence."""
    if not chunks or _below_threshold(chunks):
        return NO_EVIDENCE
    plan = plan or plan_query(question)
    fast = _deterministic_answer(plan, chunks)
    if fast is not None:
        return fast
    response = _chat(_messages(question, chunks, plan), stream=False)
    return _strip_thinking(_message_content(response))


# =========================================================
# One-call pipeline used by the Streamlit app
# =========================================================

def answer_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    university: str = "All",
    category: str = "All",
    stream: bool = False,
) -> RAGResult:
    """Plan -> retrieve -> answer. Returns the answer and the evidence used."""
    timings = {}
    start = time.perf_counter()
    plan = plan_query(question, university, category)
    chunks = retrieve(question, top_k=top_k, plan=plan)
    timings["retrieval_s"] = time.perf_counter() - start

    if not chunks and plan.courses and plan.universities and "course" in plan.topics:
        # e.g. "Does Swinburne offer a Master of Business Analytics?"
        note = "\n".join(_missing_note(plan, [])).strip()
        if note:
            timings["total_s"] = time.perf_counter() - start
            return RAGResult(note + "\n\nCheck the university's official course search to confirm.",
                             [], "structured", plan, timings)
    if not chunks or _below_threshold(chunks):
        return RAGResult(NO_EVIDENCE, [], "no_evidence", plan, timings)

    t = time.perf_counter()
    fast = _deterministic_answer(plan, chunks)
    if fast is not None:
        timings["generation_s"] = time.perf_counter() - t
        timings["total_s"] = time.perf_counter() - start
        return RAGResult(fast, chunks, "structured", plan, timings)

    if stream:
        return RAGResult("", chunks, "llm", plan, timings, stream=_stream_tokens(question, chunks, plan))

    answer = _strip_thinking(_message_content(_chat(_messages(question, chunks, plan), stream=False)))
    timings["generation_s"] = time.perf_counter() - t
    timings["total_s"] = time.perf_counter() - start
    return RAGResult(answer, chunks, "llm", plan, timings)
