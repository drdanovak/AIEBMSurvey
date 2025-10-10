import io
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Artificial Intelligence and Evidence-based Medicine: A Skills and Knowledge Survey", page_icon="🧭", layout="wide")

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

ITEMS = [
    # AILIT (4)
    ("AILIT_1", "I can explain how large language models are trained and why they sometimes hallucinate clinically plausible but false statements.", "AILIT"),
    ("AILIT_2", "I can distinguish between generative AI output and evidence synthesized from primary sources.", "AILIT"),
    ("AILIT_3", "I can identify whether an AI-generated answer includes citations that link to actual primary sources.", "AILIT"),
    ("AILIT_4", "I can name at least two reasons for failure for clinical AI (e.g., hallucination, brittleness to wording, biased recommendations).", "AILIT"),
    # VERIF (6)
    ("VERIF_1", "When AI suggests a clinical claim, I verify it against professional guidelines before acting on it.", "VERIF"),
    ("VERIF_2", "I attempt to locate and read at least the abstract of primary studies referenced by AI outputs.", "VERIF"),
    ("VERIF_3", "I log my verification steps (sources checked, dates, guideline versions) when using AI for education or patient care.", "VERIF"),
    ("VERIF_4", "I cross-check dosing and contraindications with an independent drug-information source, even if AI provides them.", "VERIF"),
    ("VERIF_5", "For high-stakes questions, I compare outputs across at least two AI tools or search engines.", "VERIF"),
    ("VERIF_6", "I look for model or content provenance (e.g., training disclosures, last update, source links) before relying on AI.", "VERIF"),
    # EQUITY (6)
    ("EQUITY_1", "I actively consider how demographic or non-clinical wording in prompts could change AI recommendations.", "EQUITY"),
    ("EQUITY_2", "I check whether evidence cited by AI represents diverse populations relevant to my local patient community.", "EQUITY"),
    ("EQUITY_3", "When generating patient materials with AI, I assess language access, readability, and cultural appropriateness.", "EQUITY"),
    ("EQUITY_4", "I can describe at least one strategy to mitigate algorithmic bias (e.g., diverse datasets, auditing, post-deployment monitoring).", "EQUITY"),
    ("EQUITY_5", "If using speech-to-text in clinical workflows, I account for the possibility of fabricated text and verify against audio or notes.", "EQUITY"),
    ("EQUITY_6", "I seek to avoid amplified disparities when using AI for triage, education, or documentation.", "EQUITY"),
    # TRUST (4)
    ("TRUST_1", "After verification, I am appropriately confident using AI-assisted synthesis to inform clinical teaching or decisions.", "TRUST"),
    ("TRUST_2", "I am comfortable disagreeing with AI when it conflicts with guidelines or the patient’s context.", "TRUST"),
    ("TRUST_3", "I can articulate uncertainty to patients when sources (including AI) disagree.", "TRUST"),
    ("TRUST_4", "When time allows, I prioritize checking primary sources rather than relying on AI by default for quick answers.", "TRUST"),
    # COMM (4)
    ("COMM_1", "I can clearly explain to a patient how I used AI as a tool in their care.", "COMM"),
    ("COMM_2", "I can diplomatically address AI-produced information a patient brings to a visit.", "COMM"),
    ("COMM_3", "I can co-create a verified patient-education handout (accurate content, appropriate reading level) with AI.", "COMM"),
    ("COMM_4", "I can discuss privacy and data-sharing implications of using consumer AI apps with patients.", "COMM"),
    # PRO (3)
    ("PRO_1", "I document AI use and verification steps in a way that a preceptor/attending can audit.", "PRO"),
    ("PRO_2", "I obtain faculty review before sharing AI-generated materials with patients.", "PRO"),
    ("PRO_3", "I understand my institution’s policies on AI use in education and patient care.", "PRO"),
    # INTENT (4)
    ("INTENT_1", "In the next month, I intend to log provenance (sources and dates) for any AI-assisted EBM product I create.", "INTENT"),
    ("INTENT_2", "I intend to run a bias check (e.g., demographic representativeness) on AI-summarized evidence I plan to use.", "INTENT"),
    ("INTENT_3", "I intend to validate AI recommendations against at least one clinical guideline source.", "INTENT"),
    ("INTENT_4", "I intend to improve my prompts to elicit sources, uncertainty, and limitations from AI systems.", "INTENT"),
]

SUBSCALES = ["AILIT", "VERIF", "EQUITY", "TRUST", "COMM", "PRO", "INTENT"]
FULL_NAMES = {
    "AILIT":  "AI-EBM Literacy",
    "VERIF":  "Verification & Provenance",
    "EQUITY": "Bias & Equity",
    "TRUST":  "Calibration & Trust",
    "COMM":   "Patient Communication",
    "PRO":    "Professional Responsibility",
    "INTENT": "Behavioral Intentions",
}
VAR2SUB = {v: s for v, _, s in ITEMS}

# ==========================
# Helpers
# ==========================
def compute_subscale_scores(responses: dict[str, int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in SUBSCALES:
        vals = [responses.get(v) for v, _, ss in ITEMS if ss == s and pd.notna(responses.get(v, np.nan))]
        out[s] = round(float(np.mean(vals)), 2) if vals else np.nan
    return out

def radar_plot(scores_now: dict[str, float]):
    cats = SUBSCALES  # chart can stay abbreviated for compactness
    r_now = [scores_now.get(c, 0) if not pd.isna(scores_now.get(c, np.nan)) else 0 for c in cats]
    cats_closed = cats + [cats[0]]
    r_now_closed = r_now + [r_now[0]]

    if PLOTLY_OK and go is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=r_now_closed, theta=cats_closed, fill="toself", name="Current", opacity=0.7))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[1, 7], tickmode="linear", dtick=1)),
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            height=540,
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
        "low":  "Watch the LLM crash-course (10–15 min) and practice explaining hallucinations and training data to a peer.",
        "ok":   "Teach-back: in 2–3 sentences, distinguish AI summaries from guideline-based synthesis.",
        "high": "Create a 1-paragraph primer for classmates on model provenance and failure modes.",
    },
    "VERIF": {
        "low":  "Pick one AI claim and verify it against a named guideline; paste the citation and version/date in your notes.",
        "ok":   "Add a simple provenance log template (Sources/Date/Guideline) to your workflow and use it once this week.",
        "high": "Run a 2-engine cross-check (AI vs. PubMed+Guideline) and note any discrepancies.",
    },
    "EQUITY": {
        "low":  "Rewrite a prompt to remove non-clinical demographic cues; check if the output changes.",
        "ok":   "Scan one AI-cited trial for population representativeness relative to your clinic.",
        "high": "Draft a 3-item bias audit checklist for your team to use on patient materials.",
    },
    "TRUST": {
        "low":  "Practice stating uncertainty to a patient using the Ask-Tell-Ask structure.",
        "ok":   "Write one ‘disagree with AI’ note tied to a guideline citation.",
        "high": "Model calibration: document one case where verification increased confidence appropriately.",
    },
    "COMM": {
        "low":  "Co-create a 6th-grade–level patient handout and get preceptor feedback.",
        "ok":   "Role-play responding to AI info a patient brings; aim for curiosity-first language.",
        "high": "Publish a short ‘How I used AI’ paragraph template for clinic notes.",
    },
    "PRO": {
        "low":  "Locate your institution’s AI policy and list two requirements in your notes.",
        "ok":   "Add an ‘AI used + verification steps’ line to your documentation template.",
        "high": "Share an anonymized example of auditable AI use with your team.",
    },
    "INTENT": {
        "low":  "Set one concrete goal (e.g., log sources for your next AI-assisted summary).",
        "ok":   "Schedule a 15-min block to run a bias check on your next case.",
        "high": "Mentor a peer through provenance logging on a mini-assignment.",
    },
}

def band_of(x: float) -> str:
    if pd.isna(x):
        return "low"  # treat missing as low for nudges
    for name, (lo, hi) in BANDS.items():
        if lo <= float(x) <= hi:
            return name
    return "low"

def make_custom_narrative(subscale_scores: dict[str, float]) -> tuple[list[str], list[str]]:
    strengths, growth = [], []
    for s in SUBSCALES:
        sc = subscale_scores.get(s, np.nan)
        b = band_of(sc)
        pretty = FULL_NAMES[s]
        if b == "high":
            strengths.append(f"{pretty} ({sc:.2f})")
        elif b == "low":
            growth.append(f"{pretty} ({'—' if pd.isna(sc) else f'{sc:.2f}'})")
    return strengths, growth

def topk(scores: dict[str, float], k=2, reverse=True):
    vals = [(s, v) for s, v in scores.items() if not pd.isna(v)]
    vals.sort(key=lambda x: x[1], reverse=reverse)
    return vals[:k]

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
    overall = np.nanmean([v for v in subscale_scores.values() if not pd.isna(v)])

    strengths, growth = make_custom_narrative(subscale_scores)
    highs = topk(subscale_scores, k=2, reverse=True)
    lows  = topk(subscale_scores, k=2, reverse=False)

    lines = []
    lines.append("AI–EBM Survey Report")
    lines.append("====================")
    lines.append(f"Timestamp (UTC): {timestamp_iso}")
    lines.append(f"Mode: {mode}")
    lines.append(f"Role: {role or '—'}")
    lines.append(f"AI Expertise: {ai_hours or '—'}")
    lines.append(f"AI Use Frequency: {ai_freq or '—'}")
    lines.append(f"Specialty: {spec or '—'}")
    lines.append(f"AI Tools Used: {ai_tools or '—'}")
    lines.append(f"Languages: {langs or '—'}")
    lines.append(f"Completion: {completion:.0f}%  |  Overall mean (1–7): {('—' if pd.isna(overall) else f'{overall:.2f}')}")
    lines.append("")

    lines.append("Subscale Scores (1–7)")
    lines.append("---------------------")
    for s in SUBSCALES:
        cur = subscale_scores.get(s, np.nan)
        cur_str = "—" if pd.isna(cur) else f"{cur:.2f}"
        lines.append(f"- {FULL_NAMES[s]}: {cur_str}")
    lines.append("")

    # Targeted narrative summary
    lines.append("Personalized Summary")
    lines.append("--------------------")
    if strengths:
        lines.append("Strengths: " + ", ".join(strengths))
    if growth:
        lines.append("Growth Areas: " + ", ".join(growth))
    if not strengths and not growth:
        lines.append("Balanced profile without clear strengths or gaps identified.")
    if highs:
        lines.append("Top areas: " + ", ".join([f"{FULL_NAMES[k]} ({v:.2f})" for k, v in highs]))
    if lows:
        lines.append("Lowest areas: " + ", ".join([f"{FULL_NAMES[k]} ({v:.2f})" for k, v in lows]))
    lines.append("")

    # Action items tailored by band (with full names)
    lines.append("Action Plan (next 1–2 weeks)")
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
    lines.append("Scores are means per subscale. 1 = Strongly disagree … 7 = Strongly agree. Lower completion may reduce score stability.")
    return "\n".join(lines)

# ==========================
# UI
# ==========================
left, right = st.columns([1, 1])
with left:
    st.title("🧭 AI-EBM Survey (Item-by-item, 1–7)")
    mode = st.radio("Survey mode", ["Pre", "Post"], horizontal=True)

# Demographics first
st.subheader("Demographics & Background")
col1, col2, col3 = st.columns(3)
with col1:
    role = st.selectbox("What is your role?", ["", "MS1", "MS2", "MS3", "MS4", "Resident", "Fellow", "Faculty", "Other"], index=0)
with col2:
    ai_hours = st.selectbox("What is your current level of AI expertise?", ["", "None", "Low", "Medium", "High", "Expert"], index=0)
with col3:
    ai_freq = st.selectbox("How often do you use AI for clinical learning/teaching?", ["", "Never", "<Monthly", "Monthly", "Weekly", "Daily or almost daily"], index=0)

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

st.subheader(f"Survey — Items {start+1}–{end} of {TOTAL_ITEMS} (1–7)")
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
        "ai_hours": ai_hours,
        "ai_freq": ai_freq,
        "specialty": spec,
        "ai_tools": ai_tools,
        "languages": langs,
        **responses,
        **{f"SCORE_{k}": v for k, v in subscale_scores.items()},
    }

    st.subheader("Subscale Web Chart (1–7)")
    fig = radar_plot(subscale_scores)

    if PLOTLY_OK and isinstance(fig, go.Figure):
        st.plotly_chart(fig, use_container_width=True)
    elif MATPLOTLIB_OK and fig is not None:
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning("No chart backend installed. Install either `plotly` (recommended) or `matplotlib` to view the radar chart.")

    with st.expander("Subscale key", expanded=False):
        st.markdown("- **AI-EBM Literacy**")
        st.markdown("- **Verification & Provenance**")
        st.markdown("- **Bias & Equity**")
        st.markdown("- **Calibration & Trust**")
        st.markdown("- **Patient Communication**")
        st.markdown("- **Professional Responsibility**")
        st.markdown("- **Behavioral Intentions**")

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
    st.text_area("Copy the text below and paste into your Canvas assignment submission:", value=report_text, height=460)
    st.download_button("Download report (.txt)", data=report_text.encode("utf-8"), file_name=f"ai_ebm_{mode.lower()}_report.txt", mime="text/plain")

    # Export CSV + chart files
    st.subheader("Export Data & Chart")
    out_df = pd.DataFrame([out_row])
    st.download_button(
        "Download results (CSV)",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name=f"ai_ebm_{mode.lower()}_results.csv",
        mime="text/csv",
    )

    pdf_bytes, pdf_mime, png_bytes, png_mime = (export_chart(fig) if fig is not None else (None, None, None, None))
    if pdf_bytes is not None:
        st.download_button("Download chart (PDF)", data=pdf_bytes, file_name="ai_ebm_chart.pdf", mime="application/pdf")
    if png_bytes is not None:
        st.download_button("Download chart (PNG)", data=png_bytes, file_name="ai_ebm_chart.png", mime="image/png")

    if (pdf_bytes is None) and (png_bytes is None):
        st.caption("To enable chart downloads, install Plotly + Kaleido (preferred) or Matplotlib.")

    st.success("Done. Your responses were scored and a customized Canvas report is ready below.")
