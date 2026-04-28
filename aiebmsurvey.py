import io
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Artificial Intelligence and Evidence-based Medicine: A Skills and Knowledge Survey",
    page_icon="🧭",
    layout="wide",
)

# ---- Streamlit rerun compatibility (new: st.rerun / old: st.experimental_rerun) ----
RERUN = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)

# ---- Optional plotting backends (Plotly preferred; Matplotlib fallback) ----
PLOTLY_OK = False
PLOTLY_PDF_OK = False
try:
    import plotly.graph_objects as go  # type: ignore

    PLOTLY_OK = True
    try:
        import kaleido  # type: ignore  # enables fig.to_image for PDF/PNG

        PLOTLY_PDF_OK = True
    except Exception:
        PLOTLY_PDF_OK = False
except Exception:
    go = None  # type: ignore

MATPLOTLIB_OK = False
try:
    import matplotlib.pyplot as plt  # type: ignore

    MATPLOTLIB_OK = True
except Exception:
    plt = None  # type: ignore

# ==========================
# Instrument Configuration
# ==========================
LIKERT7_LEGEND = "1 = Strongly disagree … 7 = Strongly agree"

# The core spider/radar chart intentionally excludes Behavioral Intentions.
# Intentions are useful implementation outcomes, but they are different from current readiness/skill.
READINESS_SUBSCALES = ["AILIT", "VERIF", "EQUITY", "TRUST", "COMM", "PRO"]
INTENTION_SUBSCALES = ["INTENT"]
SUBSCALES = READINESS_SUBSCALES + INTENTION_SUBSCALES

FULL_NAMES = {
    "AILIT": "AI-EBM Literacy",
    "VERIF": "Verification & Provenance",
    "EQUITY": "Bias & Equity",
    "TRUST": "Calibration & Trust",
    "COMM": "Patient Communication",
    "PRO": "Professional Responsibility & Workflow Safety",
    "INTENT": "Implementation Intentions",
}

SHORT_DEFINITIONS = {
    "AILIT": "Understanding how LLMs generate output, fail, and differ from evidence synthesis.",
    "VERIF": "Checking claims against guidelines, primary literature, provenance, and patient context.",
    "EQUITY": "Recognizing and reducing bias, representativeness gaps, and inequitable AI effects.",
    "TRUST": "Calibrating reliance on AI rather than over-trusting or reflexively rejecting it.",
    "COMM": "Explaining AI use, uncertainty, privacy, consent, and patient-brought AI information.",
    "PRO": "Documenting, supervising, escalating, and safely integrating AI into clinical workflows.",
    "INTENT": "Planned near-term behaviors for applying the AI-EBM workflow after the course.",
}

ITEMS = [
    # AILIT (4)
    (
        "AILIT_1",
        "I can explain, at a high level, how large language models generate text and why they sometimes produce clinically plausible but false statements.",
        "AILIT",
    ),
    (
        "AILIT_2",
        "I can distinguish between generative AI output and evidence synthesized from primary sources.",
        "AILIT",
    ),
    (
        "AILIT_3",
        "I can identify whether an AI-generated answer includes citations that link to actual primary sources.",
        "AILIT",
    ),
    (
        "AILIT_4",
        "I can name at least two ways AI-assisted clinical work can fail, such as hallucinated citations, outdated guidance, biased recommendations, or brittle responses to prompt wording.",
        "AILIT",
    ),
    # VERIF (7)
    (
        "VERIF_1",
        "I know how to verify an AI-generated clinical claim against current professional guidelines before acting on it.",
        "VERIF",
    ),
    (
        "VERIF_2",
        "I can locate primary studies referenced by AI outputs and review enough of the article to judge whether the AI accurately represented the finding.",
        "VERIF",
    ),
    (
        "VERIF_3",
        "I log my verification steps, including sources checked, dates accessed, guideline versions, and unresolved uncertainties, when using AI for education or patient care.",
        "VERIF",
    ),
    (
        "VERIF_4",
        "I cross-check high-stakes clinical details, such as dosing, contraindications, and urgent management recommendations, against independent authoritative sources.",
        "VERIF",
    ),
    (
        "VERIF_5",
        "For high-stakes questions, I know when to use an additional AI tool, search engine, guideline source, or expert consultation to test the reliability of an AI output.",
        "VERIF",
    ),
    (
        "VERIF_6",
        "I look for available provenance information, such as model or tool version, source links, date accessed, guideline version, and known update limits, before relying on AI output.",
        "VERIF",
    ),
    (
        "VERIF_7",
        "I can classify AI-generated claims as accept, modify, reject, or uncertain based on comparison with guidelines, primary literature, and patient context.",
        "VERIF",
    ),
    # EQUITY (6)
    (
        "EQUITY_1",
        "I actively consider how demographic or non-clinical wording in prompts could change AI recommendations.",
        "EQUITY",
    ),
    (
        "EQUITY_2",
        "I check whether evidence cited by AI represents diverse populations relevant to my local patient community.",
        "EQUITY",
    ),
    (
        "EQUITY_3",
        "When generating patient materials with AI, I assess language access, readability, and cultural appropriateness.",
        "EQUITY",
    ),
    (
        "EQUITY_4",
        "I can describe at least one strategy to mitigate algorithmic bias, such as diverse datasets, auditing, subgroup performance checks, or post-deployment monitoring.",
        "EQUITY",
    ),
    (
        "EQUITY_5",
        "I can identify when AI outputs may disadvantage patients because of race, language, disability, insurance status, socioeconomic status, or other non-clinical factors.",
        "EQUITY",
    ),
    (
        "EQUITY_6",
        "I can describe a concrete safeguard to reduce the risk that AI use in triage, education, or documentation amplifies disparities.",
        "EQUITY",
    ),
    # TRUST (4)
    (
        "TRUST_1",
        "After verification, I can judge when AI-assisted synthesis is sufficiently reliable to inform clinical teaching or decisions.",
        "TRUST",
    ),
    (
        "TRUST_2",
        "I am comfortable disagreeing with AI when it conflicts with guidelines or the patient’s context.",
        "TRUST",
    ),
    (
        "TRUST_3",
        "I can articulate uncertainty to patients when sources, including AI, disagree.",
        "TRUST",
    ),
    (
        "TRUST_4",
        "I can decide when primary-source verification is necessary before using AI-generated information in teaching, documentation, or patient care.",
        "TRUST",
    ),
    # COMM (5)
    (
        "COMM_1",
        "I can clearly explain to a patient how I used AI as a tool in their care.",
        "COMM",
    ),
    (
        "COMM_2",
        "I can diplomatically address AI-produced information a patient brings to a visit.",
        "COMM",
    ),
    (
        "COMM_3",
        "I can co-create a verified patient-education handout with AI that is accurate, readable, and appropriate for the patient’s context.",
        "COMM",
    ),
    (
        "COMM_4",
        "I can discuss privacy, data-sharing, consent, and opt-out implications of using consumer or clinical AI tools with patients.",
        "COMM",
    ),
    (
        "COMM_5",
        "I can distinguish between AI use that requires patient consent, AI use that requires notification, and AI use that may not require direct patient disclosure.",
        "COMM",
    ),
    # PRO (5)
    (
        "PRO_1",
        "I document AI use and verification steps in a way that a preceptor, attending, or supervisor can audit.",
        "PRO",
    ),
    (
        "PRO_2",
        "I know when AI-generated patient materials require review by a supervising clinician before use.",
        "PRO",
    ),
    (
        "PRO_3",
        "I can identify where to find my institution’s policies or guidance on AI use in education and patient care.",
        "PRO",
    ),
    (
        "PRO_4",
        "I know how to report or escalate a suspected AI-related safety issue, such as a hallucinated medication, biased recommendation, or inaccurate AI-generated note.",
        "PRO",
    ),
    (
        "PRO_5",
        "If using speech-to-text or ambient documentation in clinical workflows, I account for the possibility of fabricated or inaccurate text and verify against the encounter, audio, or notes when appropriate.",
        "PRO",
    ),
    # INTENT (4) -- scored separately from readiness
    (
        "INTENT_1",
        "In the next month, I intend to log provenance, including sources and dates, for any AI-assisted EBM product I create.",
        "INTENT",
    ),
    (
        "INTENT_2",
        "In the next month, I intend to run a bias check, such as demographic representativeness or prompt-sensitivity review, on AI-summarized evidence I plan to use.",
        "INTENT",
    ),
    (
        "INTENT_3",
        "In the next month, I intend to validate AI recommendations against at least one current clinical guideline source before using them.",
        "INTENT",
    ),
    (
        "INTENT_4",
        "In the next month, I intend to improve my prompts to elicit sources, uncertainty, limitations, and patient-context considerations from AI systems.",
        "INTENT",
    ),
]

VAR2SUB = {v: s for v, _, s in ITEMS}

# ==========================
# Helpers
# ==========================
def compute_subscale_scores(responses: dict[str, int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in SUBSCALES:
        vals = [
            responses.get(v)
            for v, _, ss in ITEMS
            if ss == s and pd.notna(responses.get(v, np.nan))
        ]
        out[s] = round(float(np.mean(vals)), 2) if vals else np.nan
    return out


def readiness_mean(subscale_scores: dict[str, float]) -> float:
    vals = [subscale_scores.get(s, np.nan) for s in READINESS_SUBSCALES]
    vals = [v for v in vals if not pd.isna(v)]
    return round(float(np.mean(vals)), 2) if vals else np.nan


def radar_plot(scores_now: dict[str, float]):
    cats = READINESS_SUBSCALES
    r_now = [scores_now.get(c, 0) if not pd.isna(scores_now.get(c, np.nan)) else 0 for c in cats]
    cats_closed = cats + [cats[0]]
    r_now_closed = r_now + [r_now[0]]

    if PLOTLY_OK and go is not None:
        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=r_now_closed,
                theta=cats_closed,
                fill="toself",
                name="Current readiness",
                opacity=0.72,
            )
        )
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[1, 7], tickmode="linear", dtick=1)),
            showlegend=False,
            margin=dict(l=20, r=20, t=35, b=20),
            height=540,
            title="AI-EBM Readiness Profile",
        )
        return fig
    elif MATPLOTLIB_OK and plt is not None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
        angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
        angles_closed = angles + [angles[0]]
        ax.set_rmax(7)
        ax.set_rticks([1, 2, 3, 4, 5, 6, 7])
        ax.set_thetagrids(np.degrees(angles), labels=cats)
        ax.plot(angles_closed, r_now_closed)
        ax.fill(angles_closed, r_now_closed, alpha=0.25)
        ax.set_title("AI-EBM Readiness Profile")
        return fig
    else:
        return None


def export_chart(fig) -> tuple[bytes | None, str | None, bytes | None, str | None]:
    """Return (pdf_bytes, pdf_mime, png_bytes, png_mime)."""
    pdf_bytes = None
    pdf_mime = None
    png_bytes = None
    png_mime = None

    if PLOTLY_OK and PLOTLY_PDF_OK and isinstance(fig, go.Figure):
        try:
            pdf_bytes = fig.to_image(format="pdf")
            pdf_mime = "application/pdf"
            png_bytes = fig.to_image(format="png", scale=2)
            png_mime = "image/png"
            return pdf_bytes, pdf_mime, png_bytes, png_mime
        except Exception:
            pass

    if MATPLOTLIB_OK and hasattr(fig, "savefig"):
        try:
            buf_pdf = io.BytesIO()
            fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
            pdf_bytes = buf_pdf.getvalue()
            pdf_mime = "application/pdf"
        except Exception:
            pdf_bytes = None
        try:
            buf_png = io.BytesIO()
            fig.savefig(buf_png, format="png", dpi=200, bbox_inches="tight")
            png_bytes = buf_png.getvalue()
            png_mime = "image/png"
        except Exception:
            png_bytes = None

    return pdf_bytes, pdf_mime, png_bytes, png_mime


# ---- Tailored messaging for the Canvas report
BANDS = {
    "high": (5.5, 7.01),
    "ok": (4.0, 5.49),
    "low": (0.0, 3.99),
}

ACTIONS = {
    "AILIT": {
        "low": "Review the LLM foundations material and practice explaining hallucinations, training/update limits, and fabricated citations to a peer.",
        "ok": "Teach-back: in 2–3 sentences, distinguish an AI-generated summary from an evidence-based synthesis grounded in primary sources.",
        "high": "Create a 1-paragraph primer for classmates on model provenance, hallucination, and common AI failure modes.",
    },
    "VERIF": {
        "low": "Pick one AI-generated clinical claim and verify it against a named guideline or primary source; record the citation, version/date, and your accept/modify/reject decision.",
        "ok": "Use a simple provenance log once this week: claim, source checked, date accessed, guideline version, and remaining uncertainty.",
        "high": "Run a claim-by-claim verification check on an AI output and note where AI agreed with, overstated, or contradicted the evidence.",
    },
    "EQUITY": {
        "low": "Rewrite a prompt to remove non-clinical demographic cues, then check whether the output changes in a way that could disadvantage a patient group.",
        "ok": "Scan one AI-cited study or guideline for population representativeness relative to your clinical setting.",
        "high": "Draft a brief bias-audit checklist your team could use before relying on AI-generated patient materials or recommendations.",
    },
    "TRUST": {
        "low": "Practice stating uncertainty and disagreement with AI using a structured sentence: what AI suggested, what evidence says, and what remains uncertain.",
        "ok": "Write one short note explaining why you would accept, modify, or reject an AI recommendation based on evidence and patient context.",
        "high": "Document one example where verification changed your level of confidence in an AI-assisted synthesis.",
    },
    "COMM": {
        "low": "Role-play responding to patient-brought AI information using curiosity-first language and a brief explanation of verification.",
        "ok": "Draft a patient-facing explanation of how AI was used, what was verified, and what privacy or consent issue matters.",
        "high": "Create a reusable script for explaining AI use, consent/notification, and opt-out options to patients.",
    },
    "PRO": {
        "low": "Identify where your institution stores AI, privacy, documentation, and reporting guidance; list two requirements or uncertainties.",
        "ok": "Add an auditable AI-use line to your workflow: tool used, purpose, sources checked, supervisor review, and escalation if needed.",
        "high": "Share an anonymized example of safe, auditable AI use with a peer or team and include what you would report if something went wrong.",
    },
    "INTENT": {
        "low": "Set one concrete implementation goal, such as logging sources for your next AI-assisted summary.",
        "ok": "Schedule a 15-minute block to run a bias or provenance check on your next AI-assisted product.",
        "high": "Mentor a peer through provenance logging or claim verification on a mini-assignment.",
    },
}


def band_of(x: float) -> str:
    if pd.isna(x):
        return "low"  # treat missing as low for nudges
    for name, (lo, hi) in BANDS.items():
        if lo <= float(x) <= hi:
            return name
    return "low"


def topk(scores: dict[str, float], subscales: list[str], k=2, reverse=True):
    vals = [(s, scores.get(s, np.nan)) for s in subscales if not pd.isna(scores.get(s, np.nan))]
    vals.sort(key=lambda x: x[1], reverse=reverse)
    return vals[:k]


def make_custom_narrative(subscale_scores: dict[str, float]) -> tuple[list[str], list[str]]:
    strengths, growth = [], []
    for s in READINESS_SUBSCALES:
        sc = subscale_scores.get(s, np.nan)
        b = band_of(sc)
        pretty = FULL_NAMES[s]
        if b == "high":
            strengths.append(f"{pretty} ({sc:.2f})")
        elif b == "low":
            growth.append(f"{pretty} ({'—' if pd.isna(sc) else f'{sc:.2f}'})")
    return strengths, growth


def make_spider_interpretation(subscale_scores: dict[str, float]) -> str:
    readiness = readiness_mean(subscale_scores)
    highs = topk(subscale_scores, READINESS_SUBSCALES, k=2, reverse=True)
    lows = topk(subscale_scores, READINESS_SUBSCALES, k=2, reverse=False)
    values = [subscale_scores.get(s, np.nan) for s in READINESS_SUBSCALES]
    values = [float(v) for v in values if not pd.isna(v)]
    spread = round(max(values) - min(values), 2) if values else np.nan
    intent = subscale_scores.get("INTENT", np.nan)

    lines = []
    lines.append("The spider diagram shows your self-rated AI-EBM readiness across six domains. Scores closer to the center indicate lower confidence or less readiness; scores closer to the outer edge indicate higher confidence or readiness. The goal is not a perfect circle or all 7s. A useful profile shows where you can safely rely on current strengths and where you should slow down, verify, ask for supervision, or practice more deliberately.")

    if not pd.isna(readiness):
        if readiness >= 5.5:
            lines.append(f"Your overall readiness profile is high ({readiness:.2f}/7). Use the diagram to identify which domains should become teaching or leadership strengths, while still maintaining verification and documentation habits.")
        elif readiness >= 4.0:
            lines.append(f"Your overall readiness profile is developing ({readiness:.2f}/7). You likely have usable baseline skills, but the lower-scoring domains are the places where structured checklists and supervision will matter most.")
        else:
            lines.append(f"Your overall readiness profile is early-stage ({readiness:.2f}/7). Treat AI outputs as provisional, use the course templates closely, and prioritize evidence verification, documentation, and supervisory review.")

    if highs:
        top_text = ", ".join([f"{FULL_NAMES[s]} ({v:.2f})" for s, v in highs])
        lines.append(f"Your strongest area(s) are {top_text}. These are the domains you can use as anchors while developing the rest of your AI-EBM workflow.")

    if lows:
        low_text = ", ".join([f"{FULL_NAMES[s]} ({v:.2f})" for s, v in lows])
        lines.append(f"Your priority growth area(s) are {low_text}. When the course asks you to verify, communicate, or document AI use, pay special attention to these domains.")

    if not pd.isna(spread):
        if spread >= 2.0:
            lines.append(f"Your profile is uneven, with a {spread:.2f}-point gap between your highest and lowest readiness domains. That kind of spiky shape means your next step is not simply to use AI more, but to balance your workflow so that strengths do not mask vulnerabilities.")
        elif spread >= 1.0:
            lines.append(f"Your profile has moderate variation, with a {spread:.2f}-point gap across domains. Look for the one or two domains that lag behind and use the action plan below to bring them closer to your stronger areas.")
        else:
            lines.append(f"Your profile is relatively balanced, with only a {spread:.2f}-point gap across domains. A balanced shape is useful, but still review whether the whole profile is high, developing, or early-stage.")

    if not pd.isna(intent):
        if intent >= 5.5:
            lines.append(f"Your Implementation Intentions score is high ({intent:.2f}/7), suggesting that you are ready to translate the course practices into near-term behavior.")
        elif intent >= 4.0:
            lines.append(f"Your Implementation Intentions score is moderate ({intent:.2f}/7). Choose one small, observable habit, such as logging sources or checking a guideline, to make the course transfer into practice.")
        else:
            lines.append(f"Your Implementation Intentions score is low ({intent:.2f}/7). Before the course ends, set one specific AI-EBM behavior you will try in the next month.")

    return "\n\n".join(lines)


def make_canvas_report(
    timestamp_iso: str,
    mode: str,
    role: str,
    ai_hours: str,
    ai_freq: str,
    spec: str,
    ai_tools: str,
    langs: str,
    subscale_scores: dict[str, float],
    responses: dict[str, int],
) -> str:
    # Completion & overall
    total_items = len(ITEMS)
    answered = sum(1 for v in responses.values() if not pd.isna(v))
    completion = 100.0 * answered / total_items if total_items else 0.0
    readiness = readiness_mean(subscale_scores)
    intent = subscale_scores.get("INTENT", np.nan)

    strengths, growth = make_custom_narrative(subscale_scores)
    highs = topk(subscale_scores, READINESS_SUBSCALES, k=2, reverse=True)
    lows = topk(subscale_scores, READINESS_SUBSCALES, k=2, reverse=False)

    lines = []
    lines.append("AI-EBM Survey Report")
    lines.append("====================")
    lines.append(f"Timestamp (UTC): {timestamp_iso}")
    lines.append(f"Mode: {mode}")
    lines.append(f"Role: {role or '—'}")
    lines.append(f"AI Expertise: {ai_hours or '—'}")
    lines.append(f"AI Use Frequency: {ai_freq or '—'}")
    lines.append(f"Specialty: {spec or '—'}")
    lines.append(f"AI Tools Used: {ai_tools or '—'}")
    lines.append(f"Languages: {langs or '—'}")
    lines.append(
        f"Completion: {completion:.0f}%  |  Readiness mean (1-7, excludes Implementation Intentions): {('—' if pd.isna(readiness) else f'{readiness:.2f}')}  |  Implementation Intentions: {('—' if pd.isna(intent) else f'{intent:.2f}') }"
    )
    lines.append("")

    lines.append("Readiness Subscale Scores (1-7)")
    lines.append("------------------------------")
    for s in READINESS_SUBSCALES:
        cur = subscale_scores.get(s, np.nan)
        cur_str = "—" if pd.isna(cur) else f"{cur:.2f}"
        lines.append(f"- {FULL_NAMES[s]}: {cur_str} — {SHORT_DEFINITIONS[s]}")
    lines.append("")

    lines.append("Implementation Intentions")
    lines.append("--------------------------")
    intent_str = "—" if pd.isna(intent) else f"{intent:.2f}"
    lines.append(f"- {FULL_NAMES['INTENT']}: {intent_str} — {SHORT_DEFINITIONS['INTENT']}")
    lines.append("")

    lines.append("How to Read Your Spider Diagram")
    lines.append("--------------------------------")
    lines.append(make_spider_interpretation(subscale_scores))
    lines.append("")

    # Targeted narrative summary
    lines.append("Personalized Summary")
    lines.append("--------------------")
    if strengths:
        lines.append("Strengths: " + ", ".join(strengths))
    if growth:
        lines.append("Growth Areas: " + ", ".join(growth))
    if not strengths and not growth:
        lines.append("Balanced profile without clear high or low domains identified.")
    if highs:
        lines.append("Top readiness areas: " + ", ".join([f"{FULL_NAMES[k]} ({v:.2f})" for k, v in highs]))
    if lows:
        lines.append("Lowest readiness areas: " + ", ".join([f"{FULL_NAMES[k]} ({v:.2f})" for k, v in lows]))
    lines.append("")

    # Action items tailored by band (with full names)
    lines.append("Action Plan (next 1-2 weeks)")
    lines.append("----------------------------")
    for s in SUBSCALES:
        sc = subscale_scores.get(s, np.nan)
        b = band_of(sc)
        lines.append(f"- {FULL_NAMES[s]}: {ACTIONS[s][b]}")
    lines.append("")

    # Item-by-item list
    lines.append("Item Responses")
    lines.append("--------------")
    for code, text, _ in ITEMS:
        val = responses.get(code, np.nan)
        val_str = "—" if pd.isna(val) else str(int(val))
        lines.append(f"- {code}: {val_str} — {text}")
    lines.append("")

    lines.append("Notes")
    lines.append("-----")
    lines.append("Scores are means per subscale. 1 = Strongly disagree … 7 = Strongly agree. The spider diagram displays six readiness domains only; Implementation Intentions are reported separately because they reflect planned future behavior rather than current skill or confidence. Lower completion may reduce score stability.")
    return "\n".join(lines)


# ==========================
# UI
# ==========================
left, right = st.columns([1, 1])
with left:
    st.title("🧭 Artificial Intelligence and Evidence-based Medicine: A Skills and Knowledge Survey")
    mode = st.radio("Survey mode", ["Pre", "Post"], horizontal=True)

with right:
    st.info(
        "This survey is a self-assessment of AI-EBM readiness. The spider diagram summarizes six readiness domains. Implementation Intentions are scored separately because they describe planned future behavior."
    )

# Demographics first
st.subheader("Demographics & Background")
col1, col2, col3 = st.columns(3)
with col1:
    role = st.selectbox(
        "What is your role?",
        ["", "MS1", "MS2", "MS3", "MS4", "Resident", "Fellow", "Faculty", "Other"],
        index=0,
    )
with col2:
    ai_hours = st.selectbox(
        "What is your current level of AI expertise?",
        ["", "None", "Low", "Medium", "High", "Expert"],
        index=0,
    )
with col3:
    ai_freq = st.selectbox(
        "How often do you use AI for clinical learning/teaching?",
        ["", "Never", "<Monthly", "Monthly", "Weekly", "Daily or almost daily"],
        index=0,
    )

colx, coly = st.columns(2)
with colx:
    spec = st.text_input("Intended/current specialty (optional)")
    ai_tools = st.text_input("Which AI tools have you used recently? (optional)")
with coly:
    langs = st.text_input("Languages (optional)")

st.divider()

# ====== Dots-only input ======
PAGE_SIZE = 7
TOTAL_ITEMS = len(ITEMS)
TOTAL_PAGES = math.ceil(TOTAL_ITEMS / PAGE_SIZE)

if "page" not in st.session_state:
    st.session_state.page = 0
if "responses" not in st.session_state:
    st.session_state.responses = {}

page = st.session_state.page
start = page * PAGE_SIZE
end = min(start + PAGE_SIZE, TOTAL_ITEMS)

st.subheader(f"Survey — Items {start + 1}–{end} of {TOTAL_ITEMS} (1–7)")
st.caption(LIKERT7_LEGEND)

# Render items (dots only)
for idx in range(start, end):
    var, text, _ = ITEMS[idx]
    current_val = st.session_state.responses.get(var)
    options = list(range(1, 8))
    default_index = (int(current_val) - 1) if isinstance(current_val, (int, np.integer)) else 3
    choice = st.radio(text, options=options, index=default_index, horizontal=True, key=f"radio_{var}")
    st.session_state.responses[var] = int(choice)

# Navigation
col_nav1, col_nav2, _ = st.columns([1, 1, 3])
with col_nav1:
    if st.button("← Back", disabled=(page == 0)):
        st.session_state.page = max(0, page - 1)
        if RERUN:
            RERUN()
with col_nav2:
    if st.button("Next →", disabled=(page >= TOTAL_PAGES - 1)):
        st.session_state.page = min(TOTAL_PAGES - 1, page + 1)
        if RERUN:
            RERUN()

# Compute
compute = st.button("Calculate, Show Chart, & Build Report ⮕")

if compute:
    timestamp_iso = datetime.utcnow().isoformat()
    responses: dict[str, int] = {v: st.session_state.responses.get(v, np.nan) for v, _, _ in ITEMS}
    subscale_scores = compute_subscale_scores(responses)

    out_row = {
        "timestamp": timestamp_iso,
        "mode": mode,
        "role": role,
        "ai_expertise": ai_hours,
        "ai_freq": ai_freq,
        "specialty": spec,
        "ai_tools": ai_tools,
        "languages": langs,
        **responses,
        **{f"SCORE_{k}": v for k, v in subscale_scores.items()},
        "SCORE_READINESS_MEAN": readiness_mean(subscale_scores),
    }

    st.subheader("Subscale Spider Diagram (1–7)")
    st.caption("The spider diagram displays the six AI-EBM readiness domains. Implementation Intentions are reported separately below.")
    fig = radar_plot(subscale_scores)

    if PLOTLY_OK and isinstance(fig, go.Figure):
        st.plotly_chart(fig, use_container_width=True)
    elif MATPLOTLIB_OK and fig is not None:
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning("No chart backend installed. Install either `plotly` (recommended) or `matplotlib` to view the spider diagram.")

    st.subheader("How to read your spider diagram")
    st.write(make_spider_interpretation(subscale_scores))

    with st.expander("Subscale key", expanded=False):
        for s in READINESS_SUBSCALES:
            st.markdown(f"- **{FULL_NAMES[s]} ({s})** — {SHORT_DEFINITIONS[s]}")
        st.markdown(f"- **{FULL_NAMES['INTENT']} (INTENT)** — {SHORT_DEFINITIONS['INTENT']} This score is not shown on the spider diagram.")

    st.subheader("Scores")
    score_df = pd.DataFrame(
        [
            {
                "Domain": FULL_NAMES[s],
                "Abbrev.": s,
                "Score": subscale_scores.get(s, np.nan),
                "Shown on spider diagram": "Yes" if s in READINESS_SUBSCALES else "No",
            }
            for s in SUBSCALES
        ]
    )
    st.dataframe(score_df, use_container_width=True, hide_index=True)

    # ===== Canvas-friendly, full-name report =====
    st.subheader("Canvas Report (copy/paste)")
    report_text = make_canvas_report(
        timestamp_iso=timestamp_iso,
        mode=mode,
        role=role,
        ai_hours=ai_hours,
        ai_freq=ai_freq,
        spec=spec,
        ai_tools=ai_tools,
        langs=langs,
        subscale_scores=subscale_scores,
        responses=responses,
    )
    st.text_area(
        "Copy the text below and paste into your Canvas assignment submission:",
        value=report_text,
        height=560,
    )
    st.download_button(
        "Download report (.txt)",
        data=report_text.encode("utf-8"),
        file_name=f"ai_ebm_{mode.lower()}_report.txt",
        mime="text/plain",
    )

    # Export CSV + chart files
    st.subheader("Export Data & Chart")
    out_df = pd.DataFrame([out_row])
    st.download_button(
        "Download results (CSV)",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name=f"ai_ebm_{mode.lower()}_results.csv",
        mime="text/csv",
    )

    pdf_bytes, pdf_mime, png_bytes, png_mime = export_chart(fig) if fig is not None else (None, None, None, None)
    if pdf_bytes is not None:
        st.download_button("Download chart (PDF)", data=pdf_bytes, file_name="ai_ebm_spider_diagram.pdf", mime="application/pdf")
    if png_bytes is not None:
        st.download_button("Download chart (PNG)", data=png_bytes, file_name="ai_ebm_spider_diagram.png", mime="image/png")

    if (pdf_bytes is None) and (png_bytes is None):
        st.caption("To enable chart downloads, install Plotly + Kaleido (preferred) or Matplotlib.")

    st.success("Done. Your responses were scored and a customized Canvas report is ready below.")
