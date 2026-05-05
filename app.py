"""
ML Preprocessing Agent — powered by OpenRouter (z-ai/glm-4.5-air:free)
Run:  streamlit run preprocessing_agent.py
Requires: pip install streamlit pandas scikit-learn matplotlib seaborn scipy python-dotenv openai tabulate
"""

import io
import json
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
)

warnings.filterwarnings("ignore")
load_dotenv()

# ─────────────────────────────────────────────────────────
# OpenRouter client
# ─────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
client = OpenAI(
    api_key=OPENROUTER_API_KEY or "placeholder",
    base_url=OPENROUTER_BASE_URL,
)

AI_ENABLED = bool(OPENROUTER_API_KEY)


def ask_deepseek(system: str, user: str, max_tokens: int = 1500) -> str:
    """Call OpenRouter chat; return text or a friendly error."""
    if not AI_ENABLED:
        return "⚠️ Set `OPENROUTER_API_KEY` in your environment to enable AI commentary."
    try:
        resp = client.chat.completions.create(
            model="z-ai/glm-4.5-air:free",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"⚠️ OpenRouter error: {exc}"


# ─────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────

def df_info_text(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()


def pretty_json(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


def fig_to_buf(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def section(title: str, icon: str = ""):
    st.markdown(f"### {icon} {title}")


# ─────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLE
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Preprocessing Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Top hero banner */
.hero {
    background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 50%, #0d1321 100%);
    border: 1px solid #2a3150;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,179,237,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #e2e8f0;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    color: #718096;
    font-size: 1.05rem;
    margin: 0;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.3);
    color: #63b3ed;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.75rem;
    letter-spacing: 1px;
}

/* Step cards */
.step-card {
    background: #13161f;
    border: 1px solid #1e2335;
    border-left: 3px solid #63b3ed;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.6rem;
}
.step-card h4 {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0 0 0.3rem 0;
}
.step-card p {
    color: #718096;
    font-size: 0.87rem;
    margin: 0;
    line-height: 1.5;
}

/* AI insight bubble */
.ai-insight {
    background: linear-gradient(135deg, #0d1321, #131929);
    border: 1px solid #2a3a5c;
    border-left: 3px solid #4299e1;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin-top: 0.8rem;
    font-size: 0.91rem;
    color: #a0aec0;
    line-height: 1.65;
}
.ai-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #4299e1;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}

/* Metric pills */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 0.8rem 0; }
.metric-pill {
    background: #1a1f2e;
    border: 1px solid #2a3150;
    border-radius: 8px;
    padding: 0.7rem 1.1rem;
    text-align: center;
    min-width: 110px;
}
.metric-pill .val {
    font-family: 'DM Mono', monospace;
    font-size: 1.3rem;
    font-weight: 500;
    color: #63b3ed;
    display: block;
}
.metric-pill .lbl {
    font-size: 0.72rem;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Step number badge */
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    background: #63b3ed;
    color: #0d1321;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 0.8rem;
    border-radius: 50%;
    margin-right: 0.5rem;
    vertical-align: middle;
}

/* Divider */
.soft-divider {
    border: none;
    border-top: 1px solid #1e2335;
    margin: 1.5rem 0;
}

/* Code-style tag */
.code-tag {
    font-family: 'DM Mono', monospace;
    background: #1a1f2e;
    color: #63b3ed;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.82rem;
}

/* Success / warning badges */
.badge-green {
    background: rgba(72,187,120,0.12);
    color: #48bb78;
    border: 1px solid rgba(72,187,120,0.25);
    border-radius: 6px;
    padding: 2px 9px;
    font-size: 0.78rem;
    font-family: 'DM Mono', monospace;
}
.badge-orange {
    background: rgba(237,137,54,0.12);
    color: #ed8936;
    border: 1px solid rgba(237,137,54,0.25);
    border-radius: 6px;
    padding: 2px 9px;
    font-size: 0.78rem;
    font-family: 'DM Mono', monospace;
}

h3 { font-family: 'Syne', sans-serif !important; color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">ML PREPROCESSING AGENT · OPENROUTER-POWERED</div>
  <h1 class="hero-title">🧬 Data Preprocessing Pipeline</h1>
  <p class="hero-sub">
    Upload any CSV — the agent runs a full 10-step ML preprocessing pipeline,
    explains every decision with AI, and exports a cleaned dataset.
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# SIDEBAR — settings
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Settings")

    missing_strategy = st.selectbox(
        "Missing value strategy",
        ["Auto (smart fill)", "Fill — mean", "Fill — median", "Fill — mode", "Drop rows"],
        index=0,
    )
    outlier_method = st.selectbox("Outlier detection method", ["IQR", "Z-score"], index=0)
    outlier_action = st.selectbox("Outlier action", ["Cap (clip)", "Remove rows"], index=0)
    z_threshold = st.slider("Z-score threshold", 2.0, 4.0, 3.0, 0.1) if outlier_method == "Z-score" else 3.0
    iqr_factor = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.25) if outlier_method == "IQR" else 1.5
    scaling_method = st.selectbox("Scaling method", ["StandardScaler", "MinMaxScaler", "Both"], index=0)
    encoding_method = st.selectbox("Categorical encoding", ["One-Hot Encoding", "Label Encoding", "Both"], index=0)
    test_size = st.slider("Test split size", 0.10, 0.40, 0.20, 0.05)
    max_onehot_unique = st.slider("Max unique values for one-hot", 2, 30, 10, 1)

    st.markdown("---")
    st.markdown("### 🔑 API Key")
    api_key_input = st.text_input("OpenRouter API Key (optional)", type="password", value="")
    if api_key_input:
        client = OpenAI(
            api_key=api_key_input,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        )
        AI_ENABLED = True

    st.markdown("---")
    st.markdown(
        "<small style='color:#4a5568;'>AI commentary requires a valid OpenRouter API key. "
        "All preprocessing runs locally with pandas + scikit-learn.</small>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Upload your CSV file",
    type=["csv"],
    help="Upload any tabular CSV dataset to begin preprocessing.",
)

if uploaded_file is None:
    st.markdown("""
    <div class="step-card">
      <h4>How it works</h4>
      <p>
        Upload a CSV → The agent runs all 10 preprocessing steps automatically →
        Get AI explanations for every decision → Download the clean dataset.
      </p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    steps_left = [
        ("1", "Load & Understand", "shape, dtypes, head, describe"),
        ("2", "Missing Values", "smart fill or drop with strategy"),
        ("3", "Duplicate Removal", "detect and drop exact duplicates"),
        ("4", "Categorical Encoding", "label encoding + one-hot encoding"),
        ("5", "Feature Scaling", "StandardScaler / MinMaxScaler"),
    ]
    steps_right = [
        ("6", "Outlier Detection", "IQR or Z-score with cap/remove"),
        ("7", "Feature Engineering", "correlation heatmap + drop low-variance"),
        ("8", "Feature Selection", "remove highly correlated features"),
        ("9", "Train/Test Split", "stratify-aware split with summary"),
        ("10", "Export", "download cleaned CSV + code snippet"),
    ]
    for num, title, desc in steps_left:
        cols[0].markdown(
            f'<div class="step-card"><h4><span class="step-num">{num}</span>{title}</h4><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )
    for num, title, desc in steps_right:
        cols[1].markdown(
            f'<div class="step-card"><h4><span class="step-num">{num}</span>{title}</h4><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )
    st.stop()


# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read file: {exc}")
    st.stop()

df = raw_df.copy()

# ─────────────────────────────────────────────────────────
# STEP 1 — Understand the dataset
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 1 — Load & Understand the Dataset", "📊")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Columns", df.shape[1])
c3.metric("Numeric cols", len(df.select_dtypes(include="number").columns))
c4.metric("Object cols", len(df.select_dtypes(include="object").columns))

tab1, tab2, tab3 = st.tabs(["📋 Head", "ℹ️ Info", "📈 Describe"])
with tab1:
    st.dataframe(df.head(10), use_container_width=True)
with tab2:
    st.code(df_info_text(df), language="text")
with tab3:
    st.dataframe(df.describe(include="all").T, use_container_width=True)

schema_summary = {
    "shape": list(df.shape),
    "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    "missing_count": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
    "unique_counts": {c: int(df[c].nunique()) for c in df.columns},
}
ai_step1 = ask_deepseek(
    "You are a senior ML engineer. Analyse this dataset summary and highlight "
    "the most important observations for preprocessing. Be concise (5-8 bullet points). "
    "Mention data types, missing values, cardinality issues, and potential preprocessing challenges.",
    f"Dataset schema:\n{pretty_json(schema_summary)}",
)
st.markdown(f'<div class="ai-label">🤖 AI ANALYSIS</div><div class="ai-insight">{ai_step1}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# STEP 2 — Missing values
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 2 — Handling Missing Values", "🩹")

missing_before = df.isnull().sum()
missing_cols = missing_before[missing_before > 0]

if missing_cols.empty:
    st.success("✅ No missing values found in this dataset.")
else:
    miss_df = pd.DataFrame({
        "Column": missing_cols.index,
        "Missing": missing_cols.values,
        "% Missing": (missing_cols.values / len(df) * 100).round(2),
        "Dtype": [str(df[c].dtype) for c in missing_cols.index],
    })
    st.dataframe(miss_df, use_container_width=True)

    # Visualise
    fig, ax = plt.subplots(figsize=(min(len(missing_cols) * 0.9 + 3, 12), 4))
    fig.patch.set_facecolor("#0d1321")
    ax.set_facecolor("#13161f")
    bars = ax.bar(missing_cols.index, missing_cols.values / len(df) * 100,
                  color="#63b3ed", edgecolor="#2a3150")
    ax.set_title("Missing Value % per Column", color="#e2e8f0", fontsize=12, pad=10)
    ax.tick_params(colors="#718096", rotation=45)
    ax.set_ylabel("% Missing", color="#718096")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2335")
    st.image(fig_to_buf(fig))

    # Apply strategy
    fill_log = {}
    for col in missing_cols.index:
        pct = missing_cols[col] / len(df)
        strat = missing_strategy

        if strat == "Drop rows" or (strat == "Auto (smart fill)" and pct > 0.50):
            df.dropna(subset=[col], inplace=True)
            fill_log[col] = "dropped rows"
        elif df[col].dtype == "object":
            fill_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col].fillna(fill_val, inplace=True)
            fill_log[col] = f"mode → {fill_val}"
        else:
            if strat in ("Fill — median", "Auto (smart fill)"):
                fill_val = df[col].median()
                df[col].fillna(fill_val, inplace=True)
                fill_log[col] = f"median → {fill_val:.4g}"
            elif strat == "Fill — mean":
                fill_val = df[col].mean()
                df[col].fillna(fill_val, inplace=True)
                fill_log[col] = f"mean → {fill_val:.4g}"
            else:
                fill_val = df[col].mode()[0]
                df[col].fillna(fill_val, inplace=True)
                fill_log[col] = f"mode → {fill_val}"

    st.markdown("**Actions taken:**")
    for col, action in fill_log.items():
        st.markdown(f'<span class="code-tag">{col}</span> &nbsp;→&nbsp; {action}', unsafe_allow_html=True)

    ai_step2 = ask_deepseek(
        "You are a senior ML engineer. Explain the missing-value handling decisions below "
        "in simple terms. Mention why median is preferred over mean for skewed data, "
        "mode for categoricals, and when to drop. Use 4–6 bullet points.",
        f"Actions taken:\n{pretty_json(fill_log)}\nDataset shape after: {df.shape}",
    )
    st.markdown(f'<div class="ai-label">🤖 AI EXPLANATION</div><div class="ai-insight">{ai_step2}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# STEP 3 — Duplicates
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 3 — Removing Duplicate Rows", "🔁")

n_dupes = df.duplicated().sum()
col1, col2 = st.columns(2)
col1.metric("Duplicate rows found", int(n_dupes))
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
col2.metric("Rows after deduplication", f"{len(df):,}")

if n_dupes > 0:
    st.markdown(f'<span class="badge-orange">Removed {n_dupes} duplicate rows</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="badge-green">No duplicates found</span>', unsafe_allow_html=True)

ai_step3 = ask_deepseek(
    "You are a senior ML engineer. Explain in 3 bullet points why removing duplicates "
    "is critical before training an ML model, and what risks arise if not done.",
    f"Duplicates removed: {n_dupes}. Remaining rows: {len(df)}",
)
st.markdown(f'<div class="ai-label">🤖 AI EXPLANATION</div><div class="ai-insight">{ai_step3}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# STEP 4 — Categorical Encoding
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 4 — Encoding Categorical Variables", "🏷️")

cat_cols = df.select_dtypes(include="object").columns.tolist()

if not cat_cols:
    st.success("✅ No categorical columns detected.")
else:
    st.markdown(f"**Detected categorical columns:** {', '.join(f'`{c}`' for c in cat_cols)}")

    le_cols, ohe_cols = [], []
    encoding_log = {}

    for col in cat_cols:
        n_unique = df[col].nunique()
        if encoding_method == "Label Encoding":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_cols.append(col)
            encoding_log[col] = f"LabelEncoded ({n_unique} classes)"
        elif encoding_method == "One-Hot Encoding":
            if n_unique <= max_onehot_unique:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                ohe_cols.append(col)
                encoding_log[col] = f"One-Hot ({n_unique} → {len(dummies.columns)} new cols)"
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                le_cols.append(col)
                encoding_log[col] = f"LabelEncoded (too many unique: {n_unique})"
        else:  # Both
            if n_unique <= max_onehot_unique:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                ohe_cols.append(col)
                encoding_log[col] = f"One-Hot ({n_unique} → {len(dummies.columns)} new cols)"
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                le_cols.append(col)
                encoding_log[col] = f"LabelEncoded (high cardinality: {n_unique})"

    enc_df = pd.DataFrame(list(encoding_log.items()), columns=["Column", "Encoding Applied"])
    st.dataframe(enc_df, use_container_width=True)
    st.metric("Columns after encoding", df.shape[1])

    ai_step4 = ask_deepseek(
        "You are a senior ML engineer. Explain the difference between label encoding and "
        "one-hot encoding, when to use each, and common interview pitfalls. "
        "Keep it to 5 bullet points, practical and interview-ready.",
        f"Encoding applied:\n{pretty_json(encoding_log)}",
    )
    st.markdown(f'<div class="ai-label">🤖 AI EXPLANATION</div><div class="ai-insight">{ai_step4}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# STEP 5 — Feature Scaling
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 5 — Feature Scaling", "📏")

num_cols = df.select_dtypes(include="number").columns.tolist()
bool_cols = [c for c in num_cols if df[c].nunique() <= 2]
scale_cols = [c for c in num_cols if c not in bool_cols]

if not scale_cols:
    st.warning("No numeric columns with >2 unique values found to scale.")
else:
    st.markdown(f"**Scaling {len(scale_cols)} numeric columns** (skipping {len(bool_cols)} binary columns)")

    df_before_scale = df[scale_cols].copy()

    if scaling_method in ("StandardScaler", "Both"):
        ss = StandardScaler()
        scaled_vals = ss.fit_transform(df[scale_cols])
        df_std = pd.DataFrame(scaled_vals, columns=[f"{c}_std" for c in scale_cols], index=df.index)

    if scaling_method in ("MinMaxScaler", "Both"):
        mms = MinMaxScaler()
        scaled_vals = mms.fit_transform(df[scale_cols])
        df_mm = pd.DataFrame(scaled_vals, columns=[f"{c}_mm" for c in scale_cols], index=df.index)

    # Apply chosen scaler to original df
    if scaling_method == "StandardScaler":
        df[scale_cols] = scaled_vals  # already done above
        df[scale_cols] = ss.transform(df_before_scale) if scaling_method == "StandardScaler" else df[scale_cols]
        ss2 = StandardScaler()
        df[scale_cols] = ss2.fit_transform(df[scale_cols])
    elif scaling_method == "MinMaxScaler":
        mms2 = MinMaxScaler()
        df[scale_cols] = mms2.fit_transform(df[scale_cols])
    else:
        ss2 = StandardScaler()
        df[scale_cols] = ss2.fit_transform(df[scale_cols])

    # Side-by-side distribution comparison (first 3 cols)
    show_cols = scale_cols[:3]
    if show_cols:
        fig, axes = plt.subplots(2, len(show_cols), figsize=(4 * len(show_cols), 6))
        if len(show_cols) == 1:
            axes = np.array(axes).reshape(2, 1)
        fig.patch.set_facecolor("#0d1321")
        for i, col in enumerate(show_cols):
            for row, (data, title) in enumerate([
                (df_before_scale[col], f"{col}\n(before)"),
                (df[col], f"{col}\n(after)"),
            ]):
                ax = axes[row][i]
                ax.set_facecolor("#13161f")
                clean_data = data.dropna().values
                bins = np.linspace(clean_data.min(), clean_data.max(), 31)
                counts, _ = np.histogram(clean_data, bins=bins)
                ax.stairs(counts, bins, color="#63b3ed" if row == 0 else "#68d391", fill=True, alpha=0.85)
                ax.set_title(title, color="#e2e8f0", fontsize=9)
                ax.tick_params(colors="#718096", labelsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#1e2335")
        fig.suptitle(f"Feature Distributions — {scaling_method}", color="#a0aec0", fontsize=11, y=1.01)
        fig.tight_layout()
        st.image(fig_to_buf(fig))

    ai_step5 = ask_deepseek(
        "You are a senior ML engineer. Explain StandardScaler vs MinMaxScaler with formulas, "
        "when to use each (e.g. SVM vs neural nets vs tree-based), and common mistakes. "
        "5–6 bullet points, interview-ready.",
        f"Scaling method applied: {scaling_method}. Columns scaled: {len(scale_cols)}",
    )
    st.markdown(f'<div class="ai-label">🤖 AI EXPLANATION</div><div class="ai-insight">{ai_step5}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# STEP 6 — Outlier Detection & Handling
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 6 — Detecting & Handling Outliers", "🎯")

num_cols_now = df.select_dtypes(include="number").columns.tolist()
outlier_log = {}
rows_before = len(df)

for col in num_cols_now:
    col_data = df[col].dropna()
    if outlier_method == "IQR":
        Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - iqr_factor * IQR, Q3 + iqr_factor * IQR
    else:
        z = np.abs(stats.zscore(col_data))
        lo = col_data[z <= z_threshold].min()
        hi = col_data[z <= z_threshold].max()

    n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
    if n_out > 0:
        if outlier_action == "Cap (clip)":
            df[col] = df[col].clip(lower=lo, upper=hi)
            outlier_log[col] = {"outliers": n_out, "action": "clipped", "lo": round(float(lo), 4), "hi": round(float(hi), 4)}
        else:
            df = df[(df[col] >= lo) & (df[col] <= hi)]
            outlier_log[col] = {"outliers": n_out, "action": "rows removed"}

rows_after = len(df)
df.reset_index(drop=True, inplace=True)

col1, col2, col3 = st.columns(3)
col1.metric("Columns with outliers", len(outlier_log))
col2.metric("Rows before", f"{rows_before:,}")
col3.metric("Rows after", f"{rows_after:,}")

if outlier_log:
    out_df = pd.DataFrame([
        {"Column": k, "Outliers": v["outliers"], "Action": v["action"]}
        for k, v in outlier_log.items()
    ])
    st.dataframe(out_df, use_container_width=True)

    # Box plots for top 4 outlier columns
    top_out_cols = sorted(outlier_log, key=lambda c: outlier_log[c]["outliers"], reverse=True)[:4]
    if top_out_cols:
        fig, axes = plt.subplots(1, len(top_out_cols), figsize=(4 * len(top_out_cols), 4))
        if len(top_out_cols) == 1:
            axes = [axes]
        fig.patch.set_facecolor("#0d1321")
        for ax, col in zip(axes, top_out_cols):
            ax.set_facecolor("#13161f")
            bp = ax.boxplot(df[col].dropna(), patch_artist=True,
                            boxprops=dict(facecolor="#1a2744", color="#63b3ed"),
                            medianprops=dict(color="#68d391", linewidth=2),
                            whiskerprops=dict(color="#718096"),
                            capprops=dict(color="#718096"),
                            flierprops=dict(marker="o", color="#fc8181", alpha=0.5, markersize=4))
            ax.set_title(col, color="#e2e8f0", fontsize=9)
            ax.tick_params(colors="#718096", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1e2335")
        fig.suptitle(f"Box Plots After {outlier_action} ({outlier_method})", color="#a0aec0", fontsize=11)
        fig.tight_layout()
        st.image(fig_to_buf(fig))
else:
    st.success("✅ No significant outliers detected.")

ai_step6 = ask_deepseek(
    "You are a senior ML engineer. Explain IQR vs Z-score outlier detection with formulas, "
    "when to cap vs remove outliers, and their effect on different ML algorithms. "
    "5–6 bullet points, practical and interview-ready.",
    f"Method: {outlier_method}. Action: {outlier_action}. Columns affected: {len(outlier_log)}",
)
st.markdown(f'<div class="ai-label">🤖 AI EXPLANATION</div><div class="ai-insight">{ai_step6}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# STEP 7 — Feature Engineering
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 7 — Feature Engineering", "⚙️")

num_cols_fe = df.select_dtypes(include="number").columns.tolist()

# Correlation heatmap
if len(num_cols_fe) >= 2:
    corr = df[num_cols_fe].corr()
    display_cols = num_cols_fe[:20]  # limit for readability
    corr_disp = corr.loc[display_cols, display_cols]
    fig_size = max(6, min(len(display_cols) * 0.65 + 2, 14))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
    fig.patch.set_facecolor("#0d1321")
    ax.set_facecolor("#0d1321")
    sns.heatmap(
        corr_disp, annot=len(display_cols) <= 12, fmt=".2f", cmap="coolwarm",
        linewidths=0.5, linecolor="#0d1321", ax=ax,
        annot_kws={"size": 7}, center=0,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Feature Correlation Heatmap", color="#e2e8f0", fontsize=12, pad=10)
    ax.tick_params(colors="#a0aec0", labelsize=8)
    st.image(fig_to_buf(fig))

# Low-variance features
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=0.0)
try:
    vt.fit(df[num_cols_fe].dropna())
    low_var = [c for c, s in zip(num_cols_fe, vt.variances_) if s == 0]
except Exception:
    low_var = []

if low_var:
    df.drop(columns=low_var, inplace=True)
    st.markdown(f"**Dropped {len(low_var)} zero-variance column(s):** {', '.join(low_var)}")
else:
    st.markdown('<span class="badge-green">No zero-variance columns found</span>', unsafe_allow_html=True)

ai_step7 = ask_deepseek(
    "You are a senior ML engineer. Explain what feature engineering involves, "
    "why correlation analysis matters, and what zero-variance features are. "
    "Also explain the risks of highly correlated features (multicollinearity). "
    "5 bullet points, interview-ready.",
    f"Numeric features: {len(num_cols_fe)}. Zero-variance dropped: {len(low_var)}",
)
st.markdown(f'<div class="ai-label">🤖 AI EXPLANATION</div><div class="ai-insight">{ai_step7}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# STEP 8 — Feature Selection (drop highly correlated)
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 8 — Feature Selection (Remove Highly Correlated)", "🔬")

num_cols_fs = df.select_dtypes(include="number").columns.tolist()
dropped_corr = []

if len(num_cols_fs) >= 2:
    corr_matrix = df[num_cols_fs].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    threshold = 0.95
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)
        dropped_corr = to_drop
        st.markdown(
            f"**Removed {len(to_drop)} highly correlated column(s)** (|r| > {threshold}): "
            + ", ".join(f"`{c}`" for c in to_drop)
        )
    else:
        st.markdown(f'<span class="badge-green">No features with |r| > {threshold} found</span>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
col1.metric("Columns before selection", len(num_cols_fs))
col2.metric("Columns after selection", df.shape[1])

ai_step8 = ask_deepseek(
    "You are a senior ML engineer. Explain feature selection techniques (filter, wrapper, embedded), "
    "why removing highly correlated features improves model performance, "
    "and mention Variance Inflation Factor (VIF). 5 bullet points, interview-ready.",
    f"Features before: {len(num_cols_fs)}. Dropped for high correlation (>0.95): {len(dropped_corr)}",
)
st.markdown(f'<div class="ai-label">🤖 AI EXPLANATION</div><div class="ai-insight">{ai_step8}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# STEP 9 — Train / Test Split
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 9 — Train / Test Split", "✂️")

n_total = len(df)
n_test = int(n_total * test_size)
n_train = n_total - n_test

# Simple index-based split (no target needed)
train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

col1, col2, col3 = st.columns(3)
col1.metric("Total samples", f"{n_total:,}")
col2.metric("Train set", f"{len(train_df):,}  ({100*(1-test_size):.0f}%)")
col3.metric("Test set", f"{len(test_df):,}  ({100*test_size:.0f}%)")

# Visual split
fig, ax = plt.subplots(figsize=(7, 1.2))
fig.patch.set_facecolor("#0d1321")
ax.set_facecolor("#0d1321")
ax.barh(0, 1 - test_size, color="#63b3ed", height=0.55, label=f"Train ({100*(1-test_size):.0f}%)")
ax.barh(0, test_size, left=1 - test_size, color="#fc8181", height=0.55, label=f"Test ({100*test_size:.0f}%)")
ax.set_xlim(0, 1)
ax.set_yticks([])
ax.set_xticks([0, 1 - test_size, 1])
ax.set_xticklabels(["0", f"{100*(1-test_size):.0f}%", "100%"], color="#718096")
for spine in ax.spines.values():
    spine.set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2,
          frameon=False, fontsize=9, labelcolor="#a0aec0")
fig.tight_layout()
st.image(fig_to_buf(fig))

ai_step9 = ask_deepseek(
    "You are a senior ML engineer. Explain train/test split best practices: "
    "why 80/20, stratified split, data leakage risks, and cross-validation. "
    "5 bullet points, interview-ready.",
    f"Split: {100*(1-test_size):.0f}/{100*test_size:.0f}. Train: {len(train_df)}, Test: {len(test_df)}",
)
st.markdown(f'<div class="ai-label">🤖 AI EXPLANATION</div><div class="ai-insight">{ai_step9}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# STEP 10 — Export cleaned dataset
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 10 — Export Cleaned Dataset", "💾")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Original rows", f"{raw_df.shape[0]:,}")
col2.metric("Cleaned rows", f"{df.shape[0]:,}")
col3.metric("Original cols", raw_df.shape[1])
col4.metric("Cleaned cols", df.shape[1])

with st.expander("📋 Preview cleaned dataset"):
    st.dataframe(df.head(20), use_container_width=True)

# Download full cleaned CSV
cleaned_csv = df.to_csv(index=False).encode("utf-8")
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

st.download_button(
    label="⬇️  Download Cleaned CSV",
    data=cleaned_csv,
    file_name=f"cleaned_dataset_{ts}.csv",
    mime="text/csv",
    type="primary",
)

# ─── Reproducible code snippet ───────────────────────────
code_snippet = f'''"""
Auto-generated preprocessing script
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats

# 1. Load
df = pd.read_csv("your_file.csv")
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())

# 2. Missing values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

# 3. Duplicates
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# 4. Encode categoricals
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    if df[col].nunique() <= {max_onehot_unique}:
        df = pd.get_dummies(df, columns=[col], drop_first=True)
    else:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# 5. Scale numerics
num_cols = df.select_dtypes(include="number").columns
scaler = StandardScaler()  # or MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 6. Outliers — IQR cap
for col in df.select_dtypes(include="number").columns:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - {iqr_factor}*IQR, Q3 + {iqr_factor}*IQR)

# 7–8. Drop zero-variance & highly correlated
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=0.0)
vt.fit(df)
df = df[df.columns[vt.get_support()]]

corr = df.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
df.drop(columns=to_drop, inplace=True)

# 9. Train/Test split
train_df, test_df = train_test_split(df, test_size={test_size}, random_state=42)
print(f"Train: {{len(train_df)}} | Test: {{len(test_df)}}")

# 10. Save
df.to_csv("cleaned_dataset.csv", index=False)
print("Saved cleaned_dataset.csv")
'''

with st.expander("🐍 View reproducible Python code"):
    st.code(code_snippet, language="python")

code_bytes = code_snippet.encode("utf-8")
st.download_button(
    label="⬇️  Download Python Script",
    data=code_bytes,
    file_name=f"preprocessing_script_{ts}.py",
    mime="text/plain",
)

# Final AI summary
ai_final = ask_deepseek(
    "You are a senior ML engineer writing an interview preparation guide. "
    "Summarise the entire preprocessing pipeline in 8 bullet points, "
    "highlighting the most common interview questions and mistakes to avoid.",
    f"""
    Pipeline completed:
    - Dataset: {raw_df.shape[0]} rows × {raw_df.shape[1]} cols  →  {df.shape[0]} rows × {df.shape[1]} cols
    - Missing strategy: {missing_strategy}
    - Outlier method: {outlier_method} ({outlier_action})
    - Encoding: {encoding_method}
    - Scaling: {scaling_method}
    - Test split: {int(test_size*100)}%
    """,
    max_tokens=1200,
)
st.markdown("---")
st.markdown("## 🎓 Interview Prep Summary")
st.markdown(f'<div class="ai-insight">{ai_final}</div>', unsafe_allow_html=True)
