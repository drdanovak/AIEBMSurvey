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
