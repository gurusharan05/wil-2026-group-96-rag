from __future__ import annotations

import re

"""Core retrieval and answer generation for the International Student RAG."""

from dataclasses import dataclass

import chromadb

import ollama

from config import (

    CHROMA_DIR,

    COLLECTION_NAME,

    DEFAULT_TOP_K,

    EMBEDDING_MODEL,

    LLM_MODEL,

    MAX_DISTANCE,

)

@dataclass

class RetrievedChunk:

    """A retrieved knowledge-base chunk."""

    text: str

    distance: float

    metadata: dict

def get_collection():

    """Connect to the existing Chroma collection.

    The collection was created without a Chroma embedding function.

    Document embeddings were generated manually using Ollama during

    ingestion, so query embeddings must also be generated manually.

    """

    client = chromadb.PersistentClient(

        path=str(CHROMA_DIR)

    )

    return client.get_collection(

        name=COLLECTION_NAME

    )

def detect_university(question: str) -> str | None:

    """Detect an explicitly mentioned university."""

    q = question.lower()

    if "rmit" in q:

        return "RMIT University"

    if (

        "university of melbourne" in q

        or "melbourne university" in q

    ):

        return "University of Melbourne"

    if "monash" in q:

        return "Monash University"

    return None

def get_mentioned_universities(

    question: str,

) -> list[str]:

    """Return all universities explicitly mentioned in a question."""

    question_lower = question.lower()

    universities = []

    if "rmit" in question_lower:

        universities.append(

            "RMIT University"

        )

    if (

        "university of melbourne" in question_lower

        or "melbourne university" in question_lower

    ):

        universities.append(

            "University of Melbourne"

        )

    if "monash" in question_lower:

        universities.append(

            "Monash University"

        )

    return universities

def _query_chroma(

    collection,

    query_embedding,

    where,

    n_results,

) -> list[RetrievedChunk]:

    """Run one Chroma query and convert results to RetrievedChunk objects."""

    results = collection.query(

        query_embeddings=[query_embedding],

        n_results=n_results,

        where=where,

    )

    documents = results.get(

        "documents",

        [[]],

    )[0]

    distances = results.get(

        "distances",

        [[]],

    )[0]

    metadatas = results.get(

        "metadatas",

        [[]],

    )[0]

    chunks = []

    for text, distance, metadata in zip(

        documents,

        distances,

        metadatas,

    ):

        chunks.append(

            RetrievedChunk(

                text=text,

                distance=float(distance),

                metadata=metadata or {},

            )

        )

    return chunks

def _retrieve_comparison_chunks(

    collection,

    question: str,

    universities: list[str],

) -> list[RetrievedChunk]:

    """Retrieve targeted course evidence for multi-university comparisons."""

    question_lower = question.lower()

    # Always retrieve the core comparison attributes. A question such as

    # "what should I consider?" may not explicitly say "fees" or "duration",

    # but those facts are still required to make the comparison complete.

    attributes = [

        ("duration", "Master of Data Science duration standard duration full-time part-time years"),

        ("location", "Master of Data Science course location campus city"),

        ("fee", "Master of Data Science international student tuition fee cost AUD"),

    ]

    if any(term in question_lower for term in [

        "entry", "admission", "requirement", "requirements", "eligibility",

        "wam", "english", "ielts", "gpa", "cricos",

    ]):

        attributes.append(

            ("requirements", "Master of Data Science entry admission eligibility requirements WAM GPA IELTS English CRICOS")

        )

    requests = []

    for university in universities:

        for attribute, query in attributes:

            requests.append({

                "university": university,

                "attribute": attribute,

                "query": f"{university} Master of Data Science {query}",

            })

    response = ollama.embed(

        model=EMBEDDING_MODEL,

        input=[request["query"] for request in requests],

    )

    selected_chunks = []

    for request, embedding in zip(requests, response["embeddings"]):

        where = {

            "$and": [

                {"university": request["university"]},

                {"category": "course"},

            ]

        }

        candidates = _query_chroma(

            collection=collection,

            query_embedding=embedding,

            where=where,

            n_results=4,

        )

        attribute = request["attribute"]

        scored = []

        for candidate in candidates:

            text = candidate.text.lower()

            section = str(candidate.metadata.get("section", "")).lower()

            score = 0

            if attribute == "location":

                if "location" in section: score += 8

                if "location" in text: score += 5

                if "campus" in text: score += 4

                if any(place in text for place in ["melbourne city", "parkville", "clayton"]): score += 4

            elif attribute == "duration":

                if "duration" in section: score += 8

                if "duration" in text: score += 5

                if "years" in text: score += 4

                if "full-time" in text: score += 2

                if "part-time" in text: score += 2

                if "1.5 years" in text: score += 2

                if "2 years" in text: score += 2

            elif attribute == "fee":

                if "fee" in section: score += 8

                if "fee" in text: score += 5

                if "tuition" in text: score += 4

                if "aud" in text: score += 3

                if "$" in text: score += 3

            elif attribute == "requirements":

                if any(term in section for term in ["entry", "admission", "requirement", "eligibility"]): score += 8

                if any(term in text for term in ["entry", "admission", "requirement", "eligibility"]): score += 5

                if any(term in text for term in ["wam", "gpa", "ielts", "english", "cricos"]): score += 3

            scored.append((score, candidate.distance, candidate))

        if scored:

            scored.sort(key=lambda item: (-item[0], item[1]))

            selected = scored[0][2]

            selected.metadata = dict(selected.metadata or {})

            selected.metadata["_comparison"] = True

            selected.metadata["_comparison_attribute"] = attribute

            selected.metadata["_comparison_university"] = request["university"]

            selected_chunks.append(selected)

    unique = {}

    for chunk in selected_chunks:

        key = (

            chunk.text,

            chunk.metadata.get("university"),

            chunk.metadata.get("section"),

            chunk.metadata.get("category"),

        )

        if key not in unique or chunk.distance < unique[key].distance:

            unique[key] = chunk

    return list(unique.values())

def retrieve(

    question: str,

    top_k: int = DEFAULT_TOP_K,

    university: str = "All",

    category: str = "All",

):

    """Topic-aware retrieval with targeted comparison support."""

    question_lower = question.lower()

    mentioned_universities = get_mentioned_universities(question)

    # Apply the Streamlit university/source filter.

    if university and university != "All":

        mentioned_universities = [university]

    implicit_university_context = any(

        phrase in question_lower

        for phrase in [

            "these universities", "these university", "choosing between these",

            "choosing between universities", "compare these universities",

            "compare these university", "compare universities", "which universities",

            "which university", "universities in melbourne", "university options",

            "university choice", "choosing a university", "choose a university",

        ]

    )

    if not mentioned_universities and implicit_university_context:

        mentioned_universities = [

            "RMIT University", "University of Melbourne", "Monash University"

        ]

    course_terms = [

        "course", "degree", "master", "data science", "duration", "fee", "fees",

        "tuition", "cost", "price", "entry", "admission", "requirement",

        "requirements", "eligibility", "wam", "cricos", "campus", "location",

    ]

    accommodation_terms = [

        "accommodation", "housing", "rent", "rental", "residence", "residential",

        "student housing", "where to live", "living",

    ]

    work_terms = [

        "work", "working", "job", "jobs", "employment", "work rights",

        "working rights", "hours", "fortnight", "part time", "part-time",

        "work while studying", "working while studying",

    ]

    topics = []

    if any(term in question_lower for term in course_terms):

        topics.append("course")

    if any(term in question_lower for term in accommodation_terms):

        topics.append("accommodation")

    if any(term in question_lower for term in work_terms):

        topics.append("work_rights")

    if not topics:

        topics = ["general"]

    # Apply the Streamlit topic filter.

    if category and category != "All":

        topics = [category]

    comparison_phrases = [

        "compare", "comparison", "choosing between", "choose between",

        "these universities", "which university", "which universities",

        "university options", "university choice",

    ]

    is_comparison = (

        len(mentioned_universities) >= 2

        and any(phrase in question_lower for phrase in comparison_phrases)

    )

    collection = get_collection()

    selected = []

    selected_keys = set()

    def chunk_key(candidate):

        return (

            candidate.text,

            candidate.metadata.get("university"),

            candidate.metadata.get("section"),

            candidate.metadata.get("category"),

        )

    def add_chunks(chunks):

        for chunk in chunks:

            key = chunk_key(chunk)

            if key not in selected_keys:

                selected.append(chunk)

                selected_keys.add(key)

    # For comparisons, retrieve separate duration/location/fee evidence for

    # every university. Do not reduce this to one generic course chunk.

    if is_comparison and "course" in topics:

        add_chunks(_retrieve_comparison_chunks(

            collection=collection,

            question=question,

            universities=mentioned_universities,

        ))

    remaining_topics = [

        topic for topic in topics

        if not (topic == "course" and is_comparison)

    ]

    requests = []

    for topic in remaining_topics:

        if topic in {"course", "accommodation"}:

            if mentioned_universities:

                for university in mentioned_universities:

                    requests.append({

                        "topic": topic,

                        "university": university,

                        "category": topic,

                        "query": f"{question} {university} {topic}",

                    })

            else:

                requests.append({

                    "topic": topic,

                    "university": None,

                    "category": topic,

                    "query": question,

                })

        elif topic == "work_rights":

            requests.append({

                "topic": topic,

                "university": None,

                "category": "work_rights",

                "query": question,

            })

        else:

            requests.append({

                "topic": topic,

                "university": None,

                "category": None,

                "query": question,

            })

    if requests:

        response = ollama.embed(

            model=EMBEDDING_MODEL,

            input=[request["query"] for request in requests],

        )

        candidates = []

        for request, embedding in zip(requests, response["embeddings"]):

            where_conditions = []

            if request["university"]:

                where_conditions.append({"university": request["university"]})

            if request["category"]:

                where_conditions.append({"category": request["category"]})

            if not where_conditions:

                where = None

            elif len(where_conditions) == 1:

                where = where_conditions[0]

            else:

                where = {"$and": where_conditions}

            results = _query_chroma(

                collection,

                embedding,

                n_results=4,

                where=where,

            )

            for result in results:

                result.metadata = dict(result.metadata or {})

                result.metadata["_topic"] = request["topic"]

                result.metadata["_requested_university"] = request["university"]

                candidates.append(result)

        deduped = {}

        for candidate in candidates:

            key = chunk_key(candidate)

            if key not in deduped or candidate.distance < deduped[key].distance:

                deduped[key] = candidate

        candidates = list(deduped.values())

        # Remove source-metadata/disclaimer chunks before selecting evidence.

        # These chunks are useful as ingestion notes but do not contain

        # substantive course, accommodation, or work-rights information.

        def is_low_value_metadata(candidate):

            metadata = candidate.metadata or {}

            section = str(metadata.get("section", "")).strip().lower()

            text = candidate.text.strip()

            if section == "rag use":

                return True

            metadata_only_markers = [

                "Important: This file is a project source snapshot prepared from the official source listed above.",

            ]

            if (

                any(marker in text for marker in metadata_only_markers)

                and len(text) < 900

            ):

                return True

            return False

        candidates = [

            candidate

            for candidate in candidates

            if not is_low_value_metadata(candidate)

        ]

        # Keep up to two accommodation chunks per university.

        if mentioned_universities:

            for topic in remaining_topics:

                if topic not in {"course", "accommodation"}:

                    continue

                for university in mentioned_universities:

                    university_candidates = [

                        candidate for candidate in candidates

                        if candidate.metadata.get("_topic") == topic

                        and (

                            candidate.metadata.get("university") == university

                            or candidate.metadata.get("_requested_university") == university

                        )

                    ]

                    university_candidates.sort(key=lambda x: x.distance)

                    add_chunks(university_candidates[:2])

        if "work_rights" in remaining_topics:

            work_candidates = [

                candidate for candidate in candidates

                if candidate.metadata.get("_topic") == "work_rights"

            ]

            work_candidates.sort(key=lambda x: x.distance)

            add_chunks(work_candidates[:2])

        if "general" in remaining_topics:

            general_candidates = [

                candidate for candidate in candidates

                if candidate.metadata.get("_topic") == "general"

            ]

            general_candidates.sort(key=lambda x: x.distance)

            add_chunks(general_candidates[:2])

        # For ordinary questions, respect top_k. Comparison questions keep all

        # targeted evidence because multiple facts per university are required.

        if not is_comparison:

            candidates.sort(key=lambda x: x.distance)

            for candidate in candidates:

                if len(selected) >= top_k:

                    break

                add_chunks([candidate])

    return selected

def _fast_fact_answer(

    question: str,

    chunks: list[RetrievedChunk],

) -> str | None:

    """

    Fast deterministic answers for structured factual questions.

    Multi-university comparisons are answered directly from the

    targeted retrieved chunks. This avoids the much slower LLM

    generation step for structured comparisons.

    """

    # =========================================================

    # MULTI-UNIVERSITY COMPARISON FAST PATH

    # =========================================================

    # ACCOMMODATION / WORK-RIGHTS FAST PATH

    # =========================================================

    question_lower = question.lower()

    accommodation_terms = [

        "accommodation", "housing", "rent", "rental", "residence",

        "where to live",

    ]

    work_rights_terms = [

        "work rights", "working rights", "work while studying",

        "work before", "work prior", "working", "work",

        "employment", "job", "hours", "fortnight",

        "workplace rights",

    ]

    is_accommodation_question = any(

        term in question_lower for term in accommodation_terms

    )

    is_work_rights_question = any(

        term in question_lower for term in work_rights_terms

    )

    # ---------------------------------------------------------

    # Accommodation fast path

    # ---------------------------------------------------------

    # Use the accommodation fast path only for accommodation-only questions.

    # Mixed course + accommodation questions must reach the grounded LLM path

    # so the answer includes both requested topics.

    course_terms_for_mixed = [

        "course", "degree", "master", "data science", "duration", "fee", "fees",

        "tuition", "cost", "entry", "admission", "requirement", "requirements",

        "eligibility", "wam", "cricos", "campus", "location",

    ]

    is_course_question = any(term in question_lower for term in course_terms_for_mixed)

    if is_accommodation_question and not is_course_question:

        university_order = [

            "RMIT University",

            "University of Melbourne",

            "Monash University",

        ]

        university_chunks = {}

        for university in university_order:

            matching_chunks = [

                chunk for chunk in chunks

                if chunk.metadata.get("university") == university

                and chunk.metadata.get("category") == "accommodation"

            ]

            if matching_chunks:

                university_chunks[university] = matching_chunks

        if university_chunks:

            answer_lines = []

            for university in university_order:

                matching_chunks = university_chunks.get(university, [])

                if not matching_chunks:

                    continue

                answer_lines.append(f"### {university}")

                added_text = set()

                for chunk in matching_chunks:

                    text = chunk.text.strip()

                    if not text or text in added_text:

                        continue

                    added_text.add(text)

                    for line in text.splitlines():

                        line = line.strip()

                        if not line:

                            continue

                        if any(line.startswith(prefix) for prefix in [

                            "Entity:", "Snapshot date:", "Source URL:",

                            "Important:", "RAG use", "Use this source"

                        ]):

                            continue

                        if line not in answer_lines:

                            answer_lines.append(f"- {line}")

                answer_lines.append("")

            if answer_lines:

                return "\n".join(answer_lines).strip()

    # ---------------------------------------------------------

    # Work-rights fast path

    # ---------------------------------------------------------

    if is_work_rights_question and not is_accommodation_question and not is_course_question:

        work_chunks = [

            chunk for chunk in chunks

            if chunk.metadata.get("category") == "work_rights"

        ]

        if work_chunks:

            answer_lines = ["### Work rights for international students"]

            added_text = set()

            for chunk in work_chunks:

                for line in chunk.text.splitlines():

                    line = line.strip()

                    if not line:

                        continue

                    if any(line.startswith(prefix) for prefix in [

                        "Entity:", "Snapshot date:", "Source URL:",

                        "Important:", "RAG use", "Use this source"

                    ]):

                        continue

                    if line in added_text:

                        continue

                    added_text.add(line)

                    answer_lines.append(f"- {line}")

            if len(answer_lines) > 1:

                return "\n".join(answer_lines).strip()

    # =========================================================

    # EXISTING MIXED-TOPIC LLM PATH

    # =========================================================

    mixed_topic_terms = [

        "accommodation", "housing", "rent", "rental", "residence",

        "working", "work rights", "work while studying", "job",

        "employment", "hours", "fortnight",

    ]

    if any(term in question_lower for term in mixed_topic_terms):

        return None

# =========================================================

    comparison_universities = []

    for chunk in chunks:

        university_name = chunk.metadata.get(

            "_comparison_university"

        )

        if (

            university_name

            and university_name not in comparison_universities

        ):

            comparison_universities.append(

                university_name

            )

    if len(comparison_universities) >= 2:

        def extract_attribute_value(

            chunk,

            attribute,

        ):

            """

            Extract the most useful value from a structured

            comparison chunk.

            """

            text = chunk.text

            lines = [

                line.strip()

                for line in text.splitlines()

                if line.strip()

            ]

            # -------------------------------------------------

            # Structured field prefixes

            # -------------------------------------------------

            prefixes = {

                "location": [

                    "Location:",

                    "Campus:",

                ],

                "duration": [

                    "Standard duration:",

                    "Full-time duration:",

                    "Duration:",

                ],

                "fee": [

                    "2027 international fee:",

                    "2026 international fee:",

                    "2027 International Fee:",

                    "2026 International Fee:",

                    "International fee:",

                    "International Fee:",

                    "International tuition fee:",

                    "International Tuition Fee:",

                    "Indicative international tuition:",

                    "Indicative International Tuition:",

                    "Tuition fee:",

                    "Tuition Fee:",

                ],

            }

            for line in lines:

                for prefix in prefixes.get(

                    attribute,

                    [],

                ):

                    if line.lower().startswith(

                        prefix.lower()

                    ):

                        value = line[

                            len(prefix):

                        ].strip()

                        if value:

                            return value

            # -------------------------------------------------

            # Location fallback

            # -------------------------------------------------

            if attribute == "location":

                patterns = [

                    r"Location:\s*([^\n]+)",

                    r"location\s*[-:]\s*([^\n]+)",

                    r"Campus:\s*([^\n]+)",

                    r"campus\s*[-:]\s*([^\n]+)",

                ]

                for pattern in patterns:

                    match = re.search(

                        pattern,

                        text,

                        re.IGNORECASE,

                    )

                    if match:

                        return match.group(

                            1

                        ).strip()

            # -------------------------------------------------

            # Duration fallback

            # -------------------------------------------------

            if attribute == "duration":

                patterns = [

                    r"Standard duration:\s*([^\n]+)",

                    r"Full-time duration:\s*([^\n]+)",

                    r"Duration:\s*([^\n]+)",

                    r"duration\s*[-:]\s*([^\n]+)",

                ]

                for pattern in patterns:

                    match = re.search(

                        pattern,

                        text,

                        re.IGNORECASE,

                    )

                    if match:

                        return match.group(

                            1

                        ).strip()

            # -------------------------------------------------

            # Fee fallback

            # -------------------------------------------------

            if attribute == "fee":

                patterns = [

                    r"2027 international fee:\s*([^\n]+)",

                    r"2026 international fee:\s*([^\n]+)",

                    r"International tuition fee:\s*([^\n]+)",

                    r"International fee:\s*([^\n]+)",

                    r"Tuition fee:\s*([^\n]+)",

                ]

                for pattern in patterns:

                    match = re.search(

                        pattern,

                        text,

                        re.IGNORECASE,

                    )

                    if match:

                        return match.group(

                            1

                        ).strip()

            return None

        # -----------------------------------------------------

        # Build comparison dictionary

        # -----------------------------------------------------

        comparison_data = {}

        for chunk in chunks:

            university_name = chunk.metadata.get(

                "_comparison_university"

            )

            attribute = chunk.metadata.get(

                "_comparison_attribute"

            )

            if not university_name:

                continue

            if not attribute:

                continue

            if university_name not in comparison_data:

                comparison_data[

                    university_name

                ] = {}

            value = extract_attribute_value(

                chunk,

                attribute,

            )

            # If structured extraction failed, retain the

            # complete chunk as a fallback.

            if value:

                comparison_data[

                    university_name

                ][attribute] = value

        # -----------------------------------------------------

        # Build final comparison answer

        # -----------------------------------------------------

        if comparison_data:

            answer_lines = []

            for university_name in comparison_universities:

                attributes = comparison_data.get(

                    university_name,

                    {},

                )

                answer_lines.append(

                    f"**{university_name}**"

                )

                # Duration

                duration = attributes.get(

                    "duration"

                )

                if duration:

                    answer_lines.append(

                        f"- **Duration:** {duration}"

                    )

                else:

                    answer_lines.append(

                        "- **Duration:** "

                        "Not explicitly stated in the retrieved evidence."

                    )

                # Location

                location = attributes.get(

                    "location"

                )

                if location:

                    answer_lines.append(

                        f"- **Location:** {location}"

                    )

                else:

                    answer_lines.append(

                        "- **Location:** "

                        "Not explicitly stated in the retrieved evidence."

                    )

                # Fee

                fee = attributes.get(

                    "fee"

                )

                if fee:

                    answer_lines.append(

                        f"- **International Fee:** {fee}"

                    )

                else:

                    answer_lines.append(

                        "- **International Fee:** "

                        "Not explicitly stated in the retrieved evidence."

                    )

                answer_lines.append("")

            return "\n".join(

                answer_lines

            ).strip()

    # =========================================================

    # EXISTING SIMPLE-FACT FAST PATH

    # =========================================================

    if not chunks:

        return None

    question_lower = question.lower()

    simple_fact_patterns = {

        "course_code": [

            "course code",

            "program code",

        ],

        "cricos": [

            "cricos",

        ],

        "fee": [

            "fee",

            "fees",

            "tuition",

            "cost",

        ],

        "duration": [

            "course duration",

            "full-time duration",

            "part-time duration",

            "duration",

            "how long",

            "how many years",

        ],

        "location": [

            "location",

            "located",

            "campus",

            "where is",

            "where the course",

            "where the program",

        ],

    }

    matched_fact_types = []

    for fact_type, patterns in simple_fact_patterns.items():

        if any(

            pattern in question_lower

            for pattern in patterns

        ):

            matched_fact_types.append(

                fact_type

            )

    if not matched_fact_types:

        return None

    chunk = chunks[0]

    lines = [

        line.strip()

        for line in chunk.text.splitlines()

        if line.strip()

    ]

    field_prefixes = {

        "course_code": [

            "Course code:",

            "Program code:",

        ],

        "cricos": [

            "CRICOS code:",

            "CRICOS:",

        ],

        "fee": [

            "International fee:",

            "2027 international fee:",

            "2026 international fee:",

            "International tuition fee:",

            "Indicative international tuition:",

            "Tuition fee:",

        ],

        "duration": [

            "Full-time duration:",

            "Standard duration:",

            "Duration:",

        ],

        "location": [

            "Location:",

            "Campus:",

        ],

    }

    extracted = {}

    for fact_type in matched_fact_types:

        prefixes = field_prefixes.get(

            fact_type,

            [],

        )

        for line in lines:

            matched_prefix = None

            for prefix in prefixes:

                if line.lower().startswith(

                    prefix.lower()

                ):

                    matched_prefix = prefix

                    break

            if matched_prefix is None:

                continue

            value = line[

                len(matched_prefix):

            ].strip()

            if value:

                extracted[fact_type] = value

                break

    if not extracted:

        return None

    answer_lines = []

    labels = {

        "course_code": "Course Code",

        "cricos": "CRICOS",

        "fee": "International Fee",

        "duration": "Duration",

        "location": "Location",

    }

    for fact_type in matched_fact_types:

        value = extracted.get(

            fact_type

        )

        if value:

            answer_lines.append(

                f"- **{labels[fact_type]}:** {value}"

            )

    if not answer_lines:

        return None

    return "\n".join(answer_lines)

def build_context(chunks):

    """

    Convert RetrievedChunk objects into the grounded context

    supplied to the LLM.

    """

    if not chunks:

        return "No relevant information was retrieved."

    sections = []

    for i, chunk in enumerate(chunks, 1):

        metadata = chunk.metadata or {}

        university = metadata.get("university", "Unknown")

        category = metadata.get("category", "Unknown")

        section = metadata.get("section", "Unknown")

        distance = getattr(chunk, "distance", None)

        header = (

            f"[Source {i}] "

            f"University: {university} | "

            f"Category: {category} | "

            f"Section: {section}"

        )

        if distance is not None:

            header += f" | Distance: {distance:.4f}"

        sections.append(

            f"{header}\n{chunk.text}"

        )

    return "\n\n".join(sections)

def generate_answer(question: str, chunks):

    """

    Generate a grounded answer using the retrieved evidence.

    The LLM must answer all requested parts of a multi-topic question

    and must inspect every supplied source before claiming information

    is unavailable.

    """

    if not chunks:

        return (

            "I could not find relevant information in the knowledge base "

            "for this question."

        )

    # ---------------------------------------------------------

    # Fast structured answer path

    # ---------------------------------------------------------

    fast_answer = _fast_fact_answer(question, chunks)

    if fast_answer is not None:

        return fast_answer

    # ---------------------------------------------------------

    # Build grounded context

    # ---------------------------------------------------------

    context = build_context(chunks)

    prompt = f"""
You are an International Student RAG assistant.

Answer the user's question using ONLY the retrieved knowledge-base evidence.

USER QUESTION:
{question}

RETRIEVED KNOWLEDGE-BASE EVIDENCE:
{context}

ANSWER RULES:
1. Answer the user's actual question directly and concisely.
2. Use only facts supported by the retrieved evidence.
3. Inspect all retrieved sources before deciding that information is unavailable.
4. Answer every part of a multi-part question.
5. Preserve numerical details exactly as stated in the evidence.
6. Do not use general knowledge to fill gaps.
7. If the evidence genuinely does not support something, say:
   "The retrieved evidence does not specify this."
8. Cite factual claims with the source labels exactly as shown, such as [Source 1].
9. Do not invent source numbers.
10. Do not recommend one university as objectively "best".

ANSWER FORMAT:
- For a simple factual question: give a short direct answer. Do not add unrelated information.
- For a course question: give only the requested course information.
- For an accommodation question: focus only on accommodation.
- For a work-rights question: focus only on work rights and workplace protections.
- For a comparison question: compare only the requested universities/topics, using a table or bullets when useful.
- For a mixed or multi-part question: answer every requested part and use clear sections when helpful.
- Do NOT automatically include accommodation, work rights, other universities, or a "What to Consider" section unless the user asks for them.

Now produce the answer.
"""

    response = ollama.chat(

        model=LLM_MODEL,

        messages=[

            {

                "role": "system",

                "content": (

                    "You are a precise, evidence-grounded RAG assistant. "

                    "Never invent facts and never omit requested parts "

                    "of a multi-part question."

                ),

            },

            {

                "role": "user",

                "content": prompt,

            },

        ],

        options={

            "temperature": 0.0,

            "think": False,

        },

    )

    return response["message"]["content"].strip()
