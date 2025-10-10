import io
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI-EBM Survey (Item-by-item Likert)", page_icon="🧭", layout="wide")

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


def radar_plot(scores_now: dict[str, float], scores_prev: dict[str, float] | None = None):
    cats = SUBSCALES
    r_now = [scores_now.get(c, 0) if not pd.isna(scores_now.get(c, np.nan)) else 0 for c in cats]
    cats_closed = cats + [cats[0]]
    r_now_closed = r_now + [r_now[0]]

    if PLOTLY_OK and go is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=r_now_closed, theta=cats_closed, fill="toself", name="Current", opacity=0.7))
        if scores_prev is not None:
            r_prev = [scores_prev.get(c, 0) if not pd.isna(scores_prev.get(c, np.nan)) else 0 for c in cats]
            r_prev_closed = r_prev + [r_prev[0]]
            fig.add_trace(go.Scatterpolar(r=r_prev_closed, theta=cats_closed, fill="toself", name="Previous", opacity=0.35))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[1, 7], tickmode="linear", dtick=1)),
            showlegend=True,
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
        ax.plot(angles_closed, r_now_closed, label="Current")
        ax.fill(angles_closed, r_now_closed, alpha=0.25)
        if scores_prev is not None:
            r_prev = [scores_prev.get(c, 0) if not pd.isna(scores_prev.get(c, np.nan)) else 0 for c in cats]
            r_prev_closed = r_prev + [r_prev[0]]
            ax.plot(angles_closed, r_prev_closed, label="Previous")
            ax.fill(angles_closed, r_prev_closed, alpha=0.15)
        ax.legend(loc="upper right")
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


def make_canvas_report(
    timestamp_iso: str,
    mode: str,
    anon_id: str,
    role: str,
    ai_hours: str,
    ai_freq: str,
    spec: str,
    ai_tools: str,
    langs: str,
    subscale_scores: dict[str, float],
    responses: dict[str, int],
    prev_scores: dict[str, float] | None = None,
) -> str:
    """Generate a copy-ready text block for Canvas."""
    # Header and demographics
    lines = []
    lines.append("AI–EBM Survey Report")
    lines.append("====================")
    lines.append(f"Timestamp (UTC): {timestamp_iso}")
    lines.append(f"Mode: {mode}")
    lines.append(f"Anonymous ID: {anon_id or '—'}")
    lines.append(f"Role: {role or '—'}")
    lines.append(f"AI Expertise: {ai_hours or '—'}")
    lines.append(f"AI Use Frequency: {ai_freq or '—'}")
    lines.append(f"Specialty: {spec or '—'}")
    lines.append(f"AI Tools Used: {ai_tools or '—'}")
    lines.append(f"Languages: {langs or '—'}")
    lines.append("")

    # Subscale scores
    lines.append("Subscale Scores (1–7)")
    lines.append("---------------------")
    for s in SUBSCALES:
        cur = subscale_scores.get(s, np.nan)
        cur_str = "—" if pd.isna(cur) else f"{cur:.2f}"
        if prev_scores is not None and s in prev_scores and not pd.isna(prev_scores[s]) and not pd.isna(cur):
            delta = cur - float(prev_scores[s])
            sign = "+" if delta > 0 else ""
            lines.append(f"- {s}: {cur_str}  ({sign}{delta:.2f} vs. previous)")
        else:
            lines.append(f"- {s}: {cur_str}")
    lines.append("")

    # Item-by-item
    lines.append("Item Responses")
    lines.append("--------------")
    for code, text, _ in ITEMS:
        val = responses.get(code, np.nan)
        val_str = "—" if pd.isna(val) else str(int(val))
        lines.append(f"- {code}: {val_str} — {text}")
    lines.append("")

    # Footer
    lines.append("Notes")
    lines.append("-----")
    lines.append("Scores are simple means per subscale. 1 = Strongly disagree … 7 = Strongly agree.")
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
    anon_id = st.text_input("Anonymous ID (optional)")
    langs = st.text_input("Languages (optional)")

st.divider()

# Input style (keep both; default Dots)
input_style = st.radio("Input style", ["Dots (1–7)", "Slider (1–7)"], horizontal=True, index=0)

# Pagination (groups of 7)
PAGE_SIZE = 7
TOTAL_ITEMS = len(ITEMS)
TOTAL_PAGES = math.ceil(TOTAL_ITEMS / PAGE_SIZE)

if "page" not in st.session_state:
    st.session_state.page = 0
if "responses" not in st.session_state:
    st.session_state.responses = {}
if "prev_scores" not in st.session_state:
    st.session_state.prev_scores = None

# Compare upload (overlay vs. prior CSV) — put BEFORE compute so it's available for plotting/exports
st.subheader("Optional: Compare with a previous attempt")
up = st.file_uploader("Upload a prior results CSV downloaded from this app (pre or post)", type=["csv"])
if up is not None:
    try:
        prev_df = pd.read_csv(up)
        score_cols = [c for c in prev_df.columns if c.startswith("SCORE_")]
        if score_cols:
            row0 = prev_df.iloc[0]
            st.session_state.prev_scores = {c.replace("SCORE_", ""): float(row0[c]) for c in score_cols if pd.notna(row0[c])}
        else:
            prev_responses = {col: int(prev_df.iloc[0][col]) for col in prev_df.columns if col in VAR2SUB}
            st.session_state.prev_scores = compute_subscale_scores(prev_responses)
        st.success("Loaded previous scores for comparison.")
    except Exception as e:
        st.warning(f"Could not parse uploaded CSV: {e}")

page = st.session_state.page
start = page * PAGE_SIZE
end = min(start + PAGE_SIZE, TOTAL_ITEMS)

st.subheader(f"Survey — Items {start+1}–{end} of {TOTAL_ITEMS} (1–7)")
st.caption(LIKERT7_LEGEND)

# Render items one-by-one (no subscale headers)
for idx in range(start, end):
    var, text, _ = ITEMS[idx]
    current_val = st.session_state.responses.get(var)

    if input_style.startswith("Dots"):
        options = list(range(1, 8))
        default_index = (int(current_val) - 1) if isinstance(current_val, (int, np.integer)) else 3
        choice = st.radio(text, options=options, index=default_index, horizontal=True, key=f"radio_{var}")
        st.session_state.responses[var] = int(choice)
    else:
        default_val = int(current_val) if isinstance(current_val, (int, np.integer)) else 4
        val = st.slider(text, min_value=1, max_value=7, step=1, value=default_val, key=f"slider_{var}")
        st.session_state.responses[var] = int(val)

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
    prev_scores = st.session_state.get("prev_scores", None)

    out_row = {
        "timestamp": timestamp_iso,
        "mode": mode,
        "anon_id": anon_id,
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
    fig = radar_plot(subscale_scores, prev_scores)

    if PLOTLY_OK and isinstance(fig, go.Figure):
        st.plotly_chart(fig, use_container_width=True)
    elif MATPLOTLIB_OK and fig is not None:
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning("No chart backend installed. Install either `plotly` (recommended) or `matplotlib` to view the radar chart.")

    with st.expander("Subscale key", expanded=False):
        st.markdown("- **AILIT** — AI-EBM Literacy")
        st.markdown("- **VERIF** — Verification & Provenance")
        st.markdown("- **EQUITY** — Bias & Equity")
        st.markdown("- **TRUST** — Calibration & Trust")
        st.markdown("- **COMM** — Patient Communication")
        st.markdown("- **PRO** — Professional Responsibility")
        st.markdown("- **INTENT** — Behavioral Intentions")

    # ===== Canvas-friendly report =====
    st.subheader("Canvas Report (copy/paste)")
    report_text = make_canvas_report(
        timestamp_iso=timestamp_iso,
        mode=mode,
        anon_id=anon_id,
        role=role,
        ai_hours=ai_hours,
        ai_freq=ai_freq,
        spec=spec,
        ai_tools=ai_tools,
        langs=langs,
        subscale_scores=subscale_scores,
        responses=responses,
        prev_scores=prev_scores,
    )
    st.text_area("Copy the text below and paste into your Canvas assignment submission:", value=report_text, height=420)
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
        st.download_button("Download chart (PDF)", data=pdf_bytes, file_name="ai_ebm_chart.pdf", mime=pdf_mime)
    if png_bytes is not None:
        st.download_button("Download chart (PNG)", data=png_bytes, file_name="ai_ebm_chart.png", mime=png_mime)

    if (pdf_bytes is None) and (png_bytes is None):
        st.caption("To enable chart downloads, install Plotly + Kaleido (preferred) or Matplotlib.")

    st.success("Done. Your item-by-item responses were scored. Use the Canvas report below or the downloads above.")
