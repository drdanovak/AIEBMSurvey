import streamlit as st
import pandas as pd
from datetime import datetime

# Optional plotting backends (Plotly preferred, fall back to Matplotlib if unavailable)
PLOTLY_OK = False
try:
    import plotly.graph_objects as go  # type: ignore
    PLOTLY_OK = True
except Exception:
    go = None  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

st.set_page_config(page_title="AI‑EBM Pre/Post Survey", page_icon="🧭", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
AGREE5 = [
    "1 – Strongly disagree",
    "2 – Disagree",
    "3 – Neutral",
    "4 – Agree",
    "5 – Strongly agree",
]
LIKELY5 = [
    "1 – Very unlikely",
    "2 – Unlikely",
    "3 – Unsure",
    "4 – Likely",
    "5 – Very likely",
]

MCQ_ALPHA = ["A", "B", "C", "D"]


def likert(key: str, text: str, scale: str = "agree5") -> int | None:
    """Render a Likert item and return a numeric 1–5 value or None."""
    lab = AGREE5 if scale == "agree5" else LIKELY5
    choice = st.radio(text, lab, index=None, key=key, horizontal=True)
    if choice is None:
        return None
    # numeric is first char before " –"
    return int(choice.split(" –")[0])


def mcq(key: str, text: str, options: list[str]) -> str | None:
    return st.radio(text, options, index=None, key=key)


def multiselect_all_that_apply(key: str, text: str, options: list[str]) -> list[str]:
    return st.multiselect(text, options, default=[], key=key)


def score_vignette(selected: list[str], correct: set[str]) -> float:
    """Exact-match scoring: 1.0 only if selected set == correct; otherwise 0."""
    return 1.0 if set(selected) == correct else 0.0


def average(values: list[int | None]) -> float | None:
    vals = [v for v in values if isinstance(v, int)]
    return round(sum(vals) / len(vals), 2) if vals else None


def spider_chart(scores: dict[str, float | None]):
    cats = list(scores.keys())
    vals = [scores[c] if scores[c] is not None else 0 for c in cats]

    # close the loop for radar
    cats_closed = cats + [cats[0]] if cats else cats
    vals_closed = vals + [vals[0]] if vals else vals

    if PLOTLY_OK and go is not None:
        fig = go.Figure(data=go.Scatterpolar(r=vals_closed, theta=cats_closed, fill="toself"))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5], tickmode="linear", dtick=1)),
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            height=520,
        )
        return fig
    else:
        # Matplotlib fallback
        import numpy as np
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        if cats:
            angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
            angles_closed = angles + [angles[0]]
            vals_closed = vals + [vals[0]]
            ax.plot(angles_closed, vals_closed)
            ax.fill(angles_closed, vals_closed, alpha=0.25)
            ax.set_thetagrids(np.degrees(angles), labels=cats)
        ax.set_rmax(5)
        ax.set_rticks([1, 2, 3, 4, 5])
        ax.grid(True)
        return fig


# -----------------------------
# Items Definition (from v2 instrument)
# -----------------------------
SURVEY = {
    "AILIT (AI‑EBM Literacy)": {
        "scale": "agree5",
        "items": [
            ("AILIT_1", "I can explain how large language models are trained and why they sometimes hallucinate clinically plausible but false statements."),
            ("AILIT_2", "I can distinguish between generative AI output and evidence synthesized from primary sources."),
            ("AILIT_3", "I can describe dataset shift and why models may not generalize to my patient population."),
            ("AILIT_4", "I can explain retrieval‑augmented generation (RAG) and how it affects source attribution and traceability."),
            ("AILIT_5", "I can identify whether an AI‑generated answer includes citations that link to actual primary sources."),
            ("AILIT_6", "I can name at least two high‑risk failure modes for clinical AI (e.g., hallucination, brittleness to wording, biased recommendations)."),
        ],
    },
    "VERIF (Verification & Provenance)": {
        "scale": "agree5",
        "items": [
            ("VERIF_1", "When AI suggests a clinical claim, I verify it against professional guidelines before acting on it."),
            ("VERIF_2", "I attempt to locate and read at least the abstract of primary studies referenced by AI outputs."),
            ("VERIF_3", "I log my verification steps (sources checked, dates, guideline versions) when using AI for education or patient care."),
            ("VERIF_4", "I cross‑check dosing and contraindications with an independent drug‑information source, even if AI provides them."),
            ("VERIF_5", "For high‑stakes questions, I compare outputs across at least two AI tools or search engines."),
            ("VERIF_6", "I look for model or content provenance (e.g., training disclosures, last update, source links) before relying on AI."),
        ],
    },
    "EQUITY (Bias & Equity)": {
        "scale": "agree5",
        "items": [
            ("EQUITY_1", "I actively consider how demographic or non‑clinical wording in prompts could change AI recommendations."),
            ("EQUITY_2", "I check whether evidence cited by AI represents diverse populations relevant to my local patient community."),
            ("EQUITY_3", "When generating patient materials with AI, I assess language access, readability, and cultural appropriateness."),
            ("EQUITY_4", "I can describe at least one strategy to mitigate algorithmic bias (e.g., diverse datasets, auditing, post‑deployment monitoring)."),
            ("EQUITY_5", "If using speech‑to‑text in clinical workflows, I account for the possibility of fabricated text and verify against audio or notes."),
            ("EQUITY_6", "I seek to avoid amplified disparities when using AI for triage, education, or documentation."),
        ],
    },
    "TRUST (Calibration & Trust)": {
        "scale": "agree5",
        "items": [
            ("TRUST_1", "After verification, I am appropriately confident using AI‑assisted synthesis to inform clinical teaching or decisions."),
            ("TRUST_2", "I am comfortable disagreeing with AI when it conflicts with guidelines or the patient’s context."),
            ("TRUST_3", "I can articulate uncertainty to patients when sources (including AI) disagree."),
            ("TRUST_4", "When time allows, I prioritize checking primary sources rather than relying on AI by default for quick answers."),
        ],
    },
    "COMM (Patient Communication)": {
        "scale": "agree5",
        "items": [
            ("COMM_1", "I can clearly explain to a patient how I used AI as a tool in their care."),
            ("COMM_2", "I can diplomatically address AI‑produced information a patient brings to a visit."),
            ("COMM_3", "I can co‑create a verified patient‑education handout (accurate content, appropriate reading level) with AI."),
            ("COMM_4", "I can discuss privacy and data‑sharing implications of using consumer AI apps with patients."),
        ],
    },
    "PRO (Professional Responsibility)": {
        "scale": "agree5",
        "items": [
            ("PRO_1", "I document AI use and verification steps in a way that a preceptor/attending can audit."),
            ("PRO_2", "I obtain faculty review before sharing AI‑generated materials with patients."),
            ("PRO_3", "I understand my institution’s policies on AI use in education and patient care."),
        ],
    },
    "INTENT (Behavioral Intentions)": {
        "scale": "likely5",
        "items": [
            ("INTENT_1", "In the next month, I intend to log provenance (sources and dates) for any AI‑assisted EBM product I create."),
            ("INTENT_2", "I intend to run a bias check (e.g., demographic representativeness) on AI‑summarized evidence I plan to use."),
            ("INTENT_3", "I intend to validate AI recommendations against at least one clinical guideline source."),
            ("INTENT_4", "I intend to improve my prompts to elicit sources, uncertainty, and limitations from AI systems."),
        ],
    },
}

KNOWLEDGE = {
    "K1": {
        "q": "Which best defines an AI ‘hallucination’?",
        "options": [
            "A – A confident, fabricated output not grounded in input data",
            "B – A minor misspelling in the output",
            "C – An accurate summary lacking citations",
            "D – A temporary server error",
        ],
        "correct": "A",
    },
    "K2": {
        "q": "What is a primary risk of relying on speech‑to‑text output in clinical contexts?",
        "options": [
            "A – It always omits filler words",
            "B – It may fabricate text that was never spoken",
            "C – It cannot handle multiple speakers",
            "D – It never timestamps audio",
        ],
        "correct": "B",
    },
    "K3": {
        "q": "Which statement about generalization is TRUE?",
        "options": [
            "A – LLMs trained on one dataset always perform equally well on all populations",
            "B – Model performance can degrade when data distributions shift",
            "C – FDA‑cleared AI tools never need verification",
            "D – Fine‑tuning eliminates the need for oversight",
        ],
        "correct": "B",
    },
    "K4": {
        "q": "Most appropriate FIRST step to verify an AI‑suggested medication dose for a patient?",
        "options": [
            "A – Ask a colleague to confirm",
            "B – Trust the AI if it cites a blog",
            "C – Check an authoritative drug database or guideline",
            "D – Search social media",
        ],
        "correct": "C",
    },
    "K5": {
        "q": "Which prompt strategy best supports traceability and safer clinical use?",
        "options": [
            "A – Ask for a single final answer only",
            "B – Ask for citations with DOIs and summaries of methods",
            "C – Prohibit the model from citing sources",
            "D – Use emojis to simplify the response",
        ],
        "correct": "B",
    },
}

VIGNETTE = {
    "key": "VT1",
    "q": "An AI summary states: ‘All adults with community‑acquired pneumonia require only 5 days of antibiotics.’ Which actions would you take before teaching this recommendation? (Select all that apply.)",
    "options": [
        "A – Check current IDSA/ATS guidelines",
        "B – Review primary trials and inclusion criteria",
        "C – Adopt the statement as‑is",
        "D – Consider comorbidities and local resistance patterns",
        "E – Log sources and dates in a provenance note",
    ],
    "correct_set": {"A", "B", "D", "E"},
}

# Post‑only (optional) — can be shown if mode == "Post"
POST_ONLY = {
    "SAT_1": ("Overall, this session improved my ability to appraise AI‑generated medical content.", "agree5"),
    "SAT_2": ("I am likely to use the ‘Prompt, But Verify’ workflow in my clinical learning or teaching.", "likely5"),
    "BARRIERS_OE": ("What barriers might prevent you from applying these practices?", "text"),
    "COMMIT_OE": ("Name one concrete change you will make in the next month.", "text"),
}

# Pre‑only (optional)
PRE_ONLY = {
    "DEM_ROLE": ("What is your role?", ["MS1", "MS2", "MS3", "MS4", "Resident", "Fellow", "Faculty", "Other"]),
    "DEM_SPECIALTY": ("Intended or current specialty/discipline (if applicable):", "text"),
    "DEM_AI_TRAIN_HRS": ("Approximate hours of prior AI/ML training (formal or self‑study):", ["None", "<5 hours", "5–20 hours", "21–50 hours", ">50 hours"]),
    "DEM_AI_FREQ": ("How often do you use AI tools for clinical learning/teaching?", ["Never", "<Monthly", "Monthly", "Weekly", "Daily or almost daily"]),
    "DEM_AI_TOOLS": ("Which AI tools have you used recently? (e.g., ChatGPT, Perplexity, Bing Copilot, Dragon/Whisper, others)", "text"),
    "DEM_EBM_TRAIN": ("Prior formal EBM training (workshops/courses):", ["None", "Introductory session(s)", "Course/elective", "Multiple courses", "Teach EBM to others"]),
    "DEM_LANGS": ("Languages you are comfortable using with patients (comma‑separated):", "text"),
}

# -----------------------------
# UI
# -----------------------------
colL, colR = st.columns([1, 2])
with colL:
    st.title("🧭 AI‑EBM Pre/Post Survey")
    st.caption("Prompt, But Verify — score your subscales and visualize progress")

    mode = st.radio("Survey mode", ["Pre", "Post"], horizontal=True)
    st.write("\n")

with colR:
    st.info(
        "This app collects responses locally in your browser session. Use the download button to export results."
    )

with st.form("survey_form", clear_on_submit=False):
    st.subheader("Subscales")

    # Render subscales
    responses: dict[str, int | None] = {}
    subscale_scores: dict[str, float | None] = {}

    for subscale, cfg in SURVEY.items():
        st.markdown(f"### {subscale}")
        vals = []
        for var, text in cfg["items"]:
            sc = likert(var, text, scale=("agree5" if cfg["scale"] == "agree5" else "likely5"))
            responses[var] = sc
            vals.append(sc)
        subscale_scores[subscale.split(" ")[0]] = average(vals)  # short tag (e.g., AILIT)
        st.divider()

    # Knowledge
    st.subheader("Knowledge & Performance")
    knowledge_answers = {}
    knowledge_score = 0
    for k, meta in KNOWLEDGE.items():
        ans = mcq(k, f"**{k}.** {meta['q']}", meta["options"])
        knowledge_answers[k] = ans
        if ans is not None:
            # convert "A – ..." to "A"
            if ans.split(" ")[0] == meta["correct"]:
                knowledge_score += 1

    vt_sel = multiselect_all_that_apply(
        VIGNETTE["key"], f"**Vignette.** {VIGNETTE['q']}", VIGNETTE["options"]
    )
    # Map selections like "A – ..." to letters
    vt_letters = {opt.split(" ")[0] for opt in vt_sel}
    vignette_score = score_vignette(list(vt_letters), VIGNETTE["correct_set"])

    # Optional sections by mode
    if mode == "Pre":
        st.subheader("Pre‑Only: Demographics & Background")
        demo = {}
        for var, (label, cfg) in PRE_ONLY.items():
            if cfg == "text":
                demo[var] = st.text_input(label, key=var)
            else:
                demo[var] = st.selectbox(label, cfg, index=None, key=var)
    else:
        st.subheader("Post‑Only: Satisfaction & Reflection")
        post = {}
        for var, (label, kind) in POST_ONLY.items():
            if kind == "text":
                post[var] = st.text_area(label, key=var)
            else:
                post[var] = likert(var, label, scale=("agree5" if kind == "agree5" else "likely5"))

    submitted = st.form_submit_button("Calculate Scores ⮕")

# -----------------------------
# Results
# -----------------------------
if submitted:
    st.success("Scores calculated.")

    # Assemble results row
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "mode": mode,
        **responses,
        **{f"SCORE_{k}": v for k, v in subscale_scores.items()},
        "KNOW_total": knowledge_score + vignette_score,
        "KNOW_mcq": knowledge_score,
        "KNOW_vignette": vignette_score,
    }

    # Display subscale table
    st.subheader("Subscale Scores (0–5)")
    score_df = pd.DataFrame(
        {
            "Subscale": list(subscale_scores.keys()),
            "Score": [v if v is not None else 0 for v in subscale_scores.values()],
        }
    )
    st.dataframe(score_df, use_container_width=True)

    # Radar chart
    st.subheader("Spider Chart of Subscales")
    fig = spider_chart(subscale_scores)
    if PLOTLY_OK and go is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.pyplot(fig, use_container_width=True)

    # Knowledge score display
    st.subheader("Knowledge & Performance Score")
    st.write(
        f"MCQ correct: **{knowledge_score}/5**  |  Vignette exact‑match: **{int(vignette_score)}/1**  |  Total: **{knowledge_score + vignette_score}/6**"
    )
    st.caption(
        "Vignette scoring is exact‑match: full credit only if you chose A, B, D, and E (and did not choose C)."
    )

    # Download
    st.subheader("Export")
    out_df = pd.DataFrame([row])
    csv = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download your results as CSV",
        data=csv,
        file_name=f"ai_ebm_{mode.lower()}_results.csv",
        mime="text/csv",
    )

    st.caption(
        "This app stores no data server‑side. For paired pre/post analysis, instruct learners to use a consistent anonymous ID in your LMS/REDCap workflow."
    )
