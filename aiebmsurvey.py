import streamlit as st
import pandas as pd
import io
from datetime import datetime

# ---- Optional plotting backends ----
PLOTLY_OK = False
PLOTLY_PDF_OK = False
try:
    import plotly.graph_objects as go  # type: ignore
    PLOTLY_OK = True
    try:
        import kaleido  # noqa: F401  # presence enables fig.to_image for PDF/PNG
        PLOTLY_PDF_OK = True
    except Exception:
        PLOTLY_PDF_OK = False
except Exception:
    go = None  # type: ignore
    PLOTLY_OK = False

import numpy as np
import matplotlib.pyplot as plt  # fallback for chart + PDF export

st.set_page_config(page_title="AI‑EBM Survey (Matrix)", page_icon="🧭", layout="wide")

# ==========================
# Instrument Configuration
# ==========================
# 1–7 Likert scale text for legend only
LIKERT7_LEGEND = "1=Strongly disagree … 7=Strongly agree"

# Item bank (NO knowledge/performance items; subscale labels are metadata only)
ITEMS = [
    # AILIT
    ("AILIT_1", "I can explain how large language models are trained and why they sometimes hallucinate clinically plausible but false statements.", "AILIT"),
    ("AILIT_2", "I can distinguish between generative AI output and evidence synthesized from primary sources.", "AILIT"),
    ("AILIT_3", "I can describe dataset shift and why models may not generalize to my patient population.", "AILIT"),
    ("AILIT_4", "I can explain retrieval‑augmented generation (RAG) and how it affects source attribution and traceability.", "AILIT"),
    ("AILIT_5", "I can identify whether an AI‑generated answer includes citations that link to actual primary sources.", "AILIT"),
    ("AILIT_6", "I can name at least two high‑risk failure modes for clinical AI (e.g., hallucination, brittleness to wording, biased recommendations).", "AILIT"),
    # VERIF
    ("VERIF_1", "When AI suggests a clinical claim, I verify it against professional guidelines before acting on it.", "VERIF"),
    ("VERIF_2", "I attempt to locate and read at least the abstract of primary studies referenced by AI outputs.", "VERIF"),
    ("VERIF_3", "I log my verification steps (sources checked, dates, guideline versions) when using AI for education or patient care.", "VERIF"),
    ("VERIF_4", "I cross‑check dosing and contraindications with an independent drug‑information source, even if AI provides them.", "VERIF"),
    ("VERIF_5", "For high‑stakes questions, I compare outputs across at least two AI tools or search engines.", "VERIF"),
    ("VERIF_6", "I look for model or content provenance (e.g., training disclosures, last update, source links) before relying on AI.", "VERIF"),
    # EQUITY
    ("EQUITY_1", "I actively consider how demographic or non‑clinical wording in prompts could change AI recommendations.", "EQUITY"),
    ("EQUITY_2", "I check whether evidence cited by AI represents diverse populations relevant to my local patient community.", "EQUITY"),
    ("EQUITY_3", "When generating patient materials with AI, I assess language access, readability, and cultural appropriateness.", "EQUITY"),
    ("EQUITY_4", "I can describe at least one strategy to mitigate algorithmic bias (e.g., diverse datasets, auditing, post‑deployment monitoring).", "EQUITY"),
    ("EQUITY_5", "If using speech‑to‑text in clinical workflows, I account for the possibility of fabricated text and verify against audio or notes.", "EQUITY"),
    ("EQUITY_6", "I seek to avoid amplified disparities when using AI for triage, education, or documentation.", "EQUITY"),
    # TRUST
    ("TRUST_1", "After verification, I am appropriately confident using AI‑assisted synthesis to inform clinical teaching or decisions.", "TRUST"),
    ("TRUST_2", "I am comfortable disagreeing with AI when it conflicts with guidelines or the patient’s context.", "TRUST"),
    ("TRUST_3", "I can articulate uncertainty to patients when sources (including AI) disagree.", "TRUST"),
    ("TRUST_4", "When time allows, I prioritize checking primary sources rather than relying on AI by default for quick answers.", "TRUST"),
    # COMM
    ("COMM_1", "I can clearly explain to a patient how I used AI as a tool in their care.", "COMM"),
    ("COMM_2", "I can diplomatically address AI‑produced information a patient brings to a visit.", "COMM"),
    ("COMM_3", "I can co‑create a verified patient‑education handout (accurate content, appropriate reading level) with AI.", "COMM"),
    ("COMM_4", "I can discuss privacy and data‑sharing implications of using consumer AI apps with patients.", "COMM"),
    # PRO
    ("PRO_1", "I document AI use and verification steps in a way that a preceptor/attending can audit.", "PRO"),
    ("PRO_2", "I obtain faculty review before sharing AI‑generated materials with patients.", "PRO"),
    ("PRO_3", "I understand my institution’s policies on AI use in education and patient care.", "PRO"),
    # INTENT
    ("INTENT_1", "In the next month, I intend to log provenance (sources and dates) for any AI‑assisted EBM product I create.", "INTENT"),
    ("INTENT_2", "I intend to run a bias check (e.g., demographic representativeness) on AI‑summarized evidence I plan to use.", "INTENT"),
    ("INTENT_3", "I intend to validate AI recommendations against at least one clinical guideline source.", "INTENT"),
    ("INTENT_4", "I intend to improve my prompts to elicit sources, uncertainty, and limitations from AI systems.", "INTENT"),
]

SUBSCALES = ["AILIT", "VERIF", "EQUITY", "TRUST", "COMM", "PRO", "INTENT"]

# mapping var -> subscale
VAR2SUB = {v: s for v, _, s in ITEMS}

# ==========================
# Helpers
# ==========================

def compute_subscale_scores(responses: dict[str, int]) -> dict[str, float]:
    out = {}
    for s in SUBSCALES:
        vals = [responses[v] for v, _, ss in ITEMS if ss == s and v in responses and pd.notna(responses[v])]
        out[s] = round(float(np.mean(vals)), 2) if vals else np.nan
    return out


def radar_plot(scores_now: dict[str, float], scores_prev: dict[str, float] | None = None):
    cats = SUBSCALES
    r_now = [scores_now.get(c, 0) if not pd.isna(scores_now.get(c, np.nan)) else 0 for c in cats]

    # Close loop
    cats_closed = cats + [cats[0]]
    r_now_closed = r_now + [r_now[0]]

    if PLOTLY_OK:
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
    else:
        # Matplotlib fallback
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


def export_chart(fig) -> tuple[bytes | None, str | None, bytes | None, str | None]:
    """Return (pdf_bytes, pdf_mime, png_bytes, png_mime). Falls back as needed."""
    pdf_bytes = None
    pdf_mime = None
    png_bytes = None
    png_mime = None

    if PLOTLY_OK and isinstance(fig, go.Figure) and PLOTLY_PDF_OK:
        try:
            pdf_bytes = fig.to_image(format="pdf")
            pdf_mime = "application/pdf"
            png_bytes = fig.to_image(format="png", scale=2)
            png_mime = "image/png"
            return pdf_bytes, pdf_mime, png_bytes, png_mime
        except Exception:
            pass

    # Matplotlib (or Plotly without kaleido) fallback
    try:
        buf_pdf = io.BytesIO()
        if hasattr(fig, "savefig"):
            fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
        else:
            # Convert Plotly to PNG in-memory via HTML rasterization not available; skip
            raise RuntimeError("No static exporter available")
        pdf_bytes = buf_pdf.getvalue()
        pdf_mime = "application/pdf"
    except Exception:
        pdf_bytes = None

    try:
        buf_png = io.BytesIO()
        if hasattr(fig, "savefig"):
            fig.savefig(buf_png, format="png", dpi=200, bbox_inches="tight")
            png_bytes = buf_png.getvalue()
            png_mime = "image/png"
    except Exception:
        png_bytes = None

    return pdf_bytes, pdf_mime, png_bytes, png_mime


# ==========================
# UI
# ==========================
left, right = st.columns([1, 1])
with left:
    st.title("🧭 AI‑EBM Survey (Matrix, 1–7)")
    st.caption("Prompt, But Verify — matrix responses, subscale scoring, overlay comparison")
    mode = st.radio("Survey mode", ["Pre", "Post"], horizontal=True)
    anon_id = st.text_input("Anonymous ID (recommended for pairing pre/post)")

with right:
    st.info("Demographics are collected first; then complete the one‑page matrix. Upload a prior CSV to compare against it.")

# ---- Demographics (front) ----
st.subheader("Demographics & Background")
col1, col2, col3 = st.columns(3)
with col1:
    role = st.selectbox("What is your role?", ["", "MS1", "MS2", "MS3", "MS4", "Resident", "Fellow", "Faculty", "Other"], index=0)
with col2:
    ai_hours = st.selectbox("Prior AI/ML training hours", ["", "None", "<5 hours", "5–20 hours", "21–50 hours", ">50 hours"], index=0)
with col3:
    ai_freq = st.selectbox("How often you use AI for clinical learning/teaching", ["", "Never", "<Monthly", "Monthly", "Weekly", "Daily or almost daily"], index=0)

spec = st.text_input("Intended/current specialty (optional)")
ai_tools = st.text_input("Which AI tools have you used recently? (optional)")
langs = st.text_input("Languages you are comfortable using with patients (optional)")

st.divider()
st.subheader("Matrix Survey (1–7)")
st.caption(LIKERT7_LEGEND)

# Build matrix DataFrame
matrix_df = pd.DataFrame({
    "Variable": [v for v, _, _ in ITEMS],
    "Item": [t for _, t, _ in ITEMS],
    "Subscale": [s for _, _, s in ITEMS],
    "Response": pd.Series([None] * len(ITEMS), dtype="Int64"),
})

# Show only Item + Response to users (no subscale titles in UI)
edited = st.data_editor(
    matrix_df[["Item", "Response"]],
    num_rows="fixed",
    use_container_width=True,
    column_config={
        "Item": st.column_config.TextColumn("Item", disabled=True, width="large"),
        "Response": st.column_config.SelectboxColumn(
            "Response (1–7)", options=list(range(1, 8)), required=True
        ),
    },
    hide_index=True,
)

# Recover responses into dict
responses = {}
for i, row in edited.reset_index(drop=True).iterrows():
    var = matrix_df.loc[i, "Variable"]
    responses[var] = int(row["Response"]) if pd.notna(row["Response"]) else np.nan

# Compare upload (for overlay vs. prior CSV)
st.subheader("Optional: Compare with a previous attempt")
up = st.file_uploader("Upload a prior results CSV downloaded from this app (pre or post)", type=["csv"])
prev_scores = None
prev_meta = {}
if up is not None:
    try:
        prev_df = pd.read_csv(up)
        # Try to read SCORE_* columns first
        score_cols = [c for c in prev_df.columns if c.startswith("SCORE_")]
        if score_cols:
            row0 = prev_df.iloc[0]
            prev_scores = {c.replace("SCORE_", ""): float(row0[c]) for c in score_cols if pd.notna(row0[c])}
            prev_meta = {k: row0.get(k) for k in ["timestamp", "mode", "anon_id"] if k in prev_df.columns}
        else:
            # If not present (older export), recompute from item responses if available
            prev_responses = {col: int(prev_df.iloc[0][col]) for col in prev_df.columns if col in VAR2SUB}
            prev_scores = compute_subscale_scores(prev_responses)
    except Exception as e:
        st.warning(f"Could not parse uploaded CSV: {e}")

# Submit & scoring
submitted = st.button("Calculate & Show Chart ⮕")

if submitted:
    # Compute scores
    subscale_scores = compute_subscale_scores(responses)

    # Build results row (no subscale table shown, per request)
    out_row = {
        "timestamp": datetime.utcnow().isoformat(),
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

    # Chart
    st.subheader("Subscale Web Chart (1–7)")
    fig = radar_plot(subscale_scores, prev_scores)

    # Render figure
    if PLOTLY_OK and isinstance(fig, go.Figure):
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.pyplot(fig, use_container_width=True)

    # Downloads
    st.subheader("Export")
    # Results CSV
    out_df = pd.DataFrame([out_row])
    st.download_button(
        "Download results (CSV)",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name=f"ai_ebm_{mode.lower()}_results.csv",
        mime="text/csv",
    )

    # Chart downloads (PDF preferred, PNG fallback)
    pdf_bytes, pdf_mime, png_bytes, png_mime = export_chart(fig)
    if pdf_bytes is not None:
        st.download_button(
            "Download chart (PDF)", data=pdf_bytes, file_name="ai_ebm_chart.pdf", mime=pdf_mime
        )
    if png_bytes is not None:
        st.download_button(
            "Download chart (PNG)", data=png_bytes, file_name="ai_ebm_chart.png", mime=png_mime
        )

    if (pdf_bytes is None) and (png_bytes is None):
        st.caption(
            "Install Plotly + Kaleido (for Plotly export) or rely on the Matplotlib fallback to enable PDF/PNG downloads."
        )

    # Light status message
    st.success("Done. Your matrix responses were scored. Use the downloads above to save data and the chart.")
