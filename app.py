"""International Student Assistant - Navy and Soft Blue UI."""

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from config import COURSE_NAMES, DEFAULT_COURSE_KEY, LLM_MODEL, STRUCTURED_DIR
from rag import IndexOutdatedError, answer_question, warm_up


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="International Student Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CUSTOM CSS - NAVY & SOFT BLUE, BALANCED ACCENTS
# =========================================================

st.markdown(
    """
    <style>
    :root {
        --page-bg: #F5F8FC;
        --card-bg: #FFFFFF;
        --accent: #294A70;
        --text: #263548;
        --muted: #65758A;
        --border: #D7E2EF;
        --soft-accent: #E8EFF7;
    }

    .stApp {
        background-color: var(--page-bg);
        color: var(--text);
    }

    /* Keep the header clear of Streamlit's toolbar */
    .block-container {
        max-width: 1200px;
        padding: 2.8rem 2rem 3rem 2rem;
    }

    h1, h2, h3 {
        color: var(--text) !important;
        font-family: Georgia, "Times New Roman", serif !important;
        letter-spacing: -0.35px;
    }

    h1 {
        font-size: 2.55rem !important;
        line-height: 1.2 !important;
        margin-top: 0.35rem !important;
    }

    h2 {
        font-size: 1.8rem !important;
    }

    h3 {
        font-size: 1.25rem !important;
    }

    p, label, li {
        color: var(--text);
    }

    .editorial-kicker {
        color: var(--accent);
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }

    .editorial-intro {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.7;
        max-width: 850px;
    }

    /* Right-aligned navigation */
    .stTabs {
        width: 100%;
    }

    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: flex-end;
        gap: 22px;
        width: 100%;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--muted);
        padding: 13px 12px;
        font-weight: 550;
        background: transparent;
    }

    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
    }

    /* Override Streamlit's default red/pink selected-tab indicator */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--accent) !important;
    }

    /* Selected university chips in the multiselect */
    [data-baseweb="tag"] {
        background-color: var(--accent) !important;
        border-color: var(--accent) !important;
        color: #FFFFFF !important;
    }

    [data-baseweb="tag"] span,
    [data-baseweb="tag"] svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    /* Buttons */
    .stButton > button,
    .stFormSubmitButton > button,
    [data-testid="stFormSubmitButton"] button {
        background-color: var(--accent) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--accent) !important;
        border-radius: 5px;
        padding: 0.55rem 1.15rem;
        font-weight: 600;
        opacity: 1 !important;
    }

    .stButton > button *,
    .stFormSubmitButton > button *,
    [data-testid="stFormSubmitButton"] button * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        background-color: #203B5A !important;
        color: #FFFFFF !important;
        border-color: #203B5A !important;
    }

    /* Text fields */
    .stTextInput input {
        background-color: #FFFFFF;
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 5px;
        min-height: 43px;
    }

    .stTextInput input:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent);
    }

    /* Selectors */
    [data-baseweb="select"] > div {
        background-color: #FFFFFF;
        border-color: var(--border);
        border-radius: 5px;
    }

    /* Cards */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
    }

    /* Fee highlights */
    [data-testid="stMetric"] {
        background-color: var(--soft-accent);
        border: 1px solid var(--border);
        border-radius: 5px;
        padding: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 0.8rem;
    }

    [data-testid="stMetricValue"] {
        color: var(--accent);
        font-size: 1.65rem;
    }

    /* Retrieved evidence */
    [data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 5px;
    }

    /* Footer */
    .editorial-footer {
        color: var(--muted);
        font-size: 0.82rem;
        text-align: center;
        line-height: 1.8;
        padding-top: 0.8rem;
    }

    hr {
        border-color: var(--border);
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1.8rem 1rem 2rem 1rem;
        }

        h1 {
            font-size: 2rem !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD STRUCTURED COURSE DATA
# =========================================================

@st.cache_data
def load_courses():
    """Load and normalise the structured university course dataset."""
    csv_path = Path(STRUCTURED_DIR) / "university_courses.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Course dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "university" in df.columns:
        df["university"] = df["university"].replace(
            {"The University of Melbourne": "University of Melbourne"}
        )
    if "course_key" not in df.columns:
        df["course_key"] = DEFAULT_COURSE_KEY
    df["course_key"] = df["course_key"].fillna(DEFAULT_COURSE_KEY)
    for column in ("verification_note", "course_code", "fee_year", "fee_basis", "intake", "campus"):
        if column in df.columns:
            df[column] = df[column].fillna("")

    return df


@st.cache_resource(show_spinner="Loading the language models (first question only)...")
def _warm_models():
    """Load the embedding model and the LLM into Ollama once per app process."""
    return warm_up()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="editorial-kicker">STUDENT PROJECT · AUSTRALIA</div>',
    unsafe_allow_html=True,
)

st.title("International Student Assistant")

st.markdown(
    """
    <div class="editorial-intro">
        Compare Master's courses (Data Science, IT, Business Analytics, Cyber Security, AI), explore university
        information, and find answers using our student knowledge base.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="
        border-top: 3px solid #294A70;
        width: 58px;
        margin-top: 22px;
        margin-bottom: 22px;
    "></div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MAIN TABS
# =========================================================

compare_tab, question_tab = st.tabs(["Compare Courses", "Ask a Question"])


# =========================================================
# TAB 1 - COMPARE COURSES
# =========================================================

with compare_tab:
    st.markdown(
        '<div class="editorial-kicker">01 / UNIVERSITY GUIDE</div>',
        unsafe_allow_html=True,
    )
    st.header("Compare Master's Courses")
    st.write("Explore course fees, duration, campus locations, and available intakes.")

    try:
        courses_df = load_courses()

        required_columns = [
            "university",
            "course_name",
            "course_code",
            "international_fee",
            "fee_year",
            "fee_basis",
            "duration_full_time",
            "campus",
            "intake",
        ]
        missing_columns = [
            column for column in required_columns if column not in courses_df.columns
        ]

        if missing_columns:
            st.error(
                "The course dataset is missing these columns: "
                + ", ".join(missing_columns)
            )
        else:
            # Build the selector from the actual CSV so newly ingested
            # universities appear automatically without editing config.py.
            available_universities = sorted(
                courses_df["university"].dropna().astype(str).str.strip().unique().tolist(),
                key=str.casefold,
            )
            available_universities = [name for name in available_universities if name]

            course_keys = [k for k in COURSE_NAMES if k in set(courses_df["course_key"])]
            selected_course = st.selectbox(
                "Course",
                options=course_keys,
                index=course_keys.index(DEFAULT_COURSE_KEY) if DEFAULT_COURSE_KEY in course_keys else 0,
                format_func=lambda key: COURSE_NAMES.get(key, key),
            )

            selected_universities = st.multiselect(
                "Select universities to compare",
                options=available_universities,
                default=available_universities,
                help="Select one or more universities.",
            )

            filtered_df = courses_df[
                courses_df["university"].isin(selected_universities)
                & (courses_df["course_key"] == selected_course)
            ].copy()

            missing = sorted(set(selected_universities) - set(filtered_df["university"]), key=str.casefold)
            if missing:
                st.caption(
                    f"No {COURSE_NAMES.get(selected_course)} record in the knowledge base for: "
                    + ", ".join(missing)
                )

            if filtered_df.empty:
                st.info("No matching courses. Select another course or more universities.")
            else:
                st.caption(f"{len(filtered_df)} course option(s) selected")

                number_of_columns = min(3, len(filtered_df))
                columns = st.columns(number_of_columns)

                for index, (_, course) in enumerate(filtered_df.iterrows()):
                    with columns[index % number_of_columns]:
                        with st.container(border=True):
                            st.markdown(
                                """
                                <div style="
                                    color: #294A70;
                                    font-size: 0.72rem;
                                    font-weight: 700;
                                    letter-spacing: 1.2px;
                                    text-transform: uppercase;
                                    margin-bottom: 8px;
                                ">UNIVERSITY</div>
                                """,
                                unsafe_allow_html=True,
                            )

                            st.subheader(course["university"])
                            st.write(f"**{course['course_name']}**")
                            st.caption(f"Course code: {course['course_code'] or 'Not stated'}")

                            fee = pd.to_numeric(
                                course["international_fee"], errors="coerce"
                            )
                            if pd.notna(fee):
                                st.metric(
                                    "Published international fee",
                                    f"AUD ${fee:,.0f}",
                                )
                            else:
                                st.metric("Published international fee", "Not captured yet")
                                st.caption(
                                    "This fee is only shown on the university's own course page. "
                                    "Use the official course page link below."
                                )

                            st.markdown("---")
                            st.markdown("**Duration**")
                            st.write(course["duration_full_time"])

                            st.markdown("**Campus**")
                            st.write(course["campus"])

                            st.markdown("**Intake**")
                            st.write(course["intake"])

                            st.markdown("---")
                            st.caption(f"Fee year: {course['fee_year'] or 'Not stated'}")
                            st.caption(f"Fee basis: {course['fee_basis']}")
                            if str(course.get("verification_note", "")).strip():
                                st.caption("⚠ Collected 19 Oct 2026, awaiting team verification")
                            if str(course.get("official_course_url", "")).strip():
                                st.markdown(f"[Official course page]({course['official_course_url']})")

                st.markdown(
                    """
                    <div style="
                        background-color: #E8EFF7;
                        border-left: 3px solid #294A70;
                        border-radius: 4px;
                        padding: 13px 16px;
                        margin-top: 16px;
                        color: #263548;
                        font-size: 0.9rem;
                        line-height: 1.6;
                    ">
                        <strong>Note about fees</strong><br>
                        Published fees may use different years and calculation
                        bases. Check the fee year and fee basis before comparing
                        total study costs.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    except FileNotFoundError as error:
        st.error(str(error))
    except Exception as error:
        st.error(
            "Unable to display the course comparison. Please check the course dataset."
        )
        st.exception(error)


# =========================================================
# TAB 2 - ASK A QUESTION
# =========================================================

with question_tab:
    st.markdown(
        '<div class="editorial-kicker">02 / KNOWLEDGE BASE</div>',
        unsafe_allow_html=True,
    )
    st.header("What would you like to know?")
    st.write(
        "Ask about courses (Data Science, IT, Business Analytics, Cyber Security, AI), tuition fees, accommodation, student work rights, "
        "or studying in Australia."
    )

    with st.container(border=True):
        st.markdown("### Ask your question")
        st.caption(
            "Ask about one university or compare multiple universities in a single question."
        )

        with st.form("question_form"):
            question = st.text_input(
                "Your question",
                placeholder="e.g. What is the RMIT Master of IT fee?",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Find answer →")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question before submitting.")
        else:
            try:
                started = time.perf_counter()
                with st.spinner("Searching the knowledge base..."):
                    # Load both Ollama models once per app process so the
                    # first question does not pay the model-load cost twice.
                    _warm_models()
                    result = answer_question(question, top_k=5, stream=True)

                st.markdown(
                    '<div class="editorial-kicker">YOUR RESULTS</div>',
                    unsafe_allow_html=True,
                )
                st.subheader("Answer")

                with st.container(border=True):
                    if result.stream is not None:
                        # Show the LLM answer token by token instead of
                        # waiting for the full response.
                        st.write_stream(result.stream)
                    else:
                        st.markdown(result.answer)

                elapsed = time.perf_counter() - started
                if result.route == "structured":
                    how = "answered directly from the knowledge-base records (no LLM call needed)"
                elif result.route == "llm":
                    how = f"generated by {LLM_MODEL} from the evidence below"
                else:
                    how = "no matching evidence found"
                st.caption(f"Answered in {elapsed:.1f}s · {how}")

                if result.evidence:
                    st.markdown("---")
                    st.subheader("Evidence used")
                    st.caption(
                        "Only the passages used for this answer are shown. "
                        "Expand a source to inspect it."
                    )

                    for index, chunk in enumerate(result.evidence, start=1):
                        metadata = getattr(chunk, "metadata", {}) or {}
                        university_name = metadata.get("university", "General information")
                        category = str(metadata.get("category", "")).replace("_", " ")
                        if metadata.get("doc_type") == "structured_course":
                            label = "Structured course record"
                        else:
                            label = metadata.get("source_file", "Knowledge-base source")

                        with st.expander(
                            f"Source {index} · {university_name} · {category} · {label}"
                        ):
                            st.text(chunk.text)

                            details = []
                            if metadata.get("source_url"):
                                details.append(f"Official page: {metadata['source_url']}")
                            if metadata.get("snapshot_date"):
                                details.append(f"Snapshot / verified: {metadata['snapshot_date']}")
                            if metadata.get("_retrieval") == "direct":
                                details.append("Matched by university + topic filter")
                            elif chunk.distance is not None:
                                details.append(f"Semantic distance: {chunk.distance:.3f}")
                            if details:
                                st.caption(" · ".join(details))

            except IndexOutdatedError as error:
                st.error(str(error))
            except Exception as error:
                st.error(
                    "Something went wrong while processing your question. "
                    "Check that Ollama is running, the required models are installed, "
                    "and ChromaDB is available."
                )
                st.exception(error)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown(
    """
    <div class="editorial-footer">
        <strong>International Student Assistant</strong><br>
        A student-built university information project<br>
        Answers are based on available knowledge-base evidence.
        Verify important information with official university and government sources.
    </div>
    """,
    unsafe_allow_html=True,
)
