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
    
    # AI Auto-configure button
    if 'auto_configured' not in st.session_state:
        st.session_state.auto_configured = False
    
    if st.button("🤖 AI Auto-Configure", use_container_width=True, type="primary"):
        st.session_state.auto_configured = True
        st.rerun()

    missing_strategy = st.selectbox(
        "Missing value strategy",
        ["Auto (smart fill)", "Fill — mean", "Fill — median", "Fill — mode", "Drop rows"],
        index=0,
        key="missing_strategy"
    )
    outlier_method = st.selectbox("Outlier detection method", ["IQR", "Z-score"], index=0, key="outlier_method")
    outlier_action = st.selectbox("Outlier action", ["Cap (clip)", "Remove rows"], index=0, key="outlier_action")
    z_threshold = st.slider("Z-score threshold", 2.0, 4.0, 3.0, 0.1, key="z_threshold") if outlier_method == "Z-score" else 3.0
    iqr_factor = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.25, key="iqr_factor") if outlier_method == "IQR" else 1.5
    scaling_method = st.selectbox("Scaling method", ["StandardScaler", "MinMaxScaler", "Both"], index=0, key="scaling_method")
    encoding_method = st.selectbox("Categorical encoding", ["One-Hot Encoding", "Label Encoding", "Both"], index=0, key="encoding_method")
    test_size = st.slider("Test split size", 0.10, 0.40, 0.20, 0.05, key="test_size")
    max_onehot_unique = st.slider("Max unique values for one-hot", 2, 30, 10, 1, key="max_onehot_unique")

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
      <h4>📋 Complete EDA & Feature Engineering Pipeline</h4>
      <p>
        Upload CSV → AI-powered analysis with <strong>inline visualizations at each step</strong> →
        Download cleaned dataset + reproducible code
      </p>
      <p><strong>⚠️ Box plots for outlier detection in Step 3!</strong></p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    steps_left = [
        ("1", "Data Understanding", "shape, types + bar plot, missing heatmap"),
        ("2", "Data Cleaning", "missing, duplicates + before/after charts"),
        ("3", "Univariate + Outliers", "distributions + BOX PLOTS (IQR)"),
        ("4", "Bivariate Analysis", "correlations + heatmap, scatter plots"),
        ("5", "Feature Transformation", "scaling, encoding + distributions"),
    ]
    steps_right = [
        ("6", "Feature Creation", "new features + distribution plots"),
        ("7", "Feature Selection", "variance, correlation + importance"),
        ("8", "Imbalanced Data", "class distribution + balancing"),
        ("9", "Train/Test Split", "stratified + distribution comparison"),
        ("10", "Export & Pipeline", "save dataset + final summary"),
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
# DROP ID-LIKE COLUMNS (High Cardinality)
# ─────────────────────────────────────────────────────────
id_columns = []
for col in df.columns:
    unique_ratio = df[col].nunique() / len(df)
    # Drop if >95% unique values (likely ID column)
    if unique_ratio > 0.95:
        id_columns.append(col)

if id_columns:
    df.drop(columns=id_columns, inplace=True)
    st.info(f"🗑️ Dropped ID-like columns (>95% unique): {', '.join(id_columns)}")

# ─────────────────────────────────────────────────────────
# WORKFLOW OVERVIEW
# ─────────────────────────────────────────────────────────
st.markdown("## 🔄 Complete EDA & Feature Engineering Workflow")
st.markdown("""
**⚠️ Important: All visualizations are generated INLINE within each step!**

**Steps 1-4: Exploratory Data Analysis (EDA)**
1. Data Understanding (+ bar plot, missing heatmap)
2. Data Cleaning (+ before/after charts)
3. Univariate Analysis + **BOX PLOTS for Outlier Detection (IQR method)**
4. Bivariate Analysis (+ correlation heatmap, scatter plots)

**Steps 5-10: Feature Engineering & ML Preparation**
5. Feature Transformation (+ before/after distributions)
6. Feature Creation (+ new feature plots)
7. Feature Selection (+ importance plots)
8. Imbalanced Data Handling (+ class distribution)
9. Train/Test Split (+ distribution comparison)
10. Export & Pipeline

📊 **Every step includes relevant visualizations inline!**
""")

# ─────────────────────────────────────────────────────────
# AI AUTO-CONFIGURATION
# ─────────────────────────────────────────────────────────
if st.session_state.get('auto_configured', False) and AI_ENABLED:
    with st.spinner("🤖 AI is analyzing your data and configuring optimal settings..."):
        data_summary = f"""
Shape: {df.shape[0]} rows × {df.shape[1]} cols
Numeric: {len(df.select_dtypes(include='number').columns)} | Categorical: {len(df.select_dtypes(include='object').columns)}
Missing: {df.isnull().sum().sum()} total in {list(df.columns[df.isnull().any()])}
Dtypes: {dict(df.dtypes.head(10))}
Unique counts: {dict(df.nunique().head(10))}
"""
        
        ai_config = ask_deepseek(
            system="You are a data scientist expert in EDA and Feature Engineering. Analyze the dataset and recommend optimal preprocessing settings following the complete data science workflow. Return ONLY a JSON object.",
            user=f"""Analyze this dataset and recommend preprocessing configuration:

{data_summary}

Follow this workflow:
1. DATA UNDERSTANDING: Analyze structure, types, missing values
2. DATA CLEANING: Recommend missing value strategy, outlier handling
3. FEATURE ENGINEERING: Suggest encoding, scaling methods
4. FEATURE SELECTION: Recommend variance threshold, correlation handling
5. TRAIN/TEST SPLIT: Suggest optimal split ratio

Return JSON with these exact keys:
- missing_strategy: "Auto (smart fill)" | "Fill — mean" | "Fill — median" | "Fill — mode" | "Drop rows"
- outlier_method: "IQR" | "Z-score"
- outlier_action: "Cap (clip)" | "Remove rows"
- z_threshold: 2.0-4.0
- iqr_factor: 1.0-3.0
- scaling_method: "StandardScaler" | "MinMaxScaler" | "Both"
- encoding_method: "One-Hot Encoding" | "Label Encoding" | "Both"
- test_size: 0.1-0.4
- max_onehot_unique: 2-30
- reasoning: brief explanation of workflow decisions

Decision Rules:
- Use mean for normally distributed data, median for skewed
- IQR for small datasets (<1000), Z-score for large (>1000)
- Cap outliers if <5% of data, remove if >5%
- StandardScaler for ML models, MinMaxScaler for neural networks
- One-Hot for <10 unique categories, Label for >10
- Test size: 0.2 for balanced, 0.3 for imbalanced

Return ONLY the JSON object.""",
            max_tokens=500
        )
        
        try:
            config = json.loads(ai_config.strip().replace('```json', '').replace('```', ''))
            
            st.session_state.missing_strategy = config['missing_strategy']
            st.session_state.outlier_method = config['outlier_method']
            st.session_state.outlier_action = config['outlier_action']
            st.session_state.z_threshold = float(config['z_threshold'])
            st.session_state.iqr_factor = float(config['iqr_factor'])
            st.session_state.scaling_method = config['scaling_method']
            st.session_state.encoding_method = config['encoding_method']
            st.session_state.test_size = float(config['test_size'])
            st.session_state.max_onehot_unique = int(config['max_onehot_unique'])
            
            st.success("✅ AI configured preprocessing pipeline")
            with st.expander("🧠 AI Reasoning"):
                st.write(config['reasoning'])
            
            st.session_state.auto_configured = False
            st.rerun()
        except Exception as e:
            st.warning(f"⚠️ AI config failed: {e}. Using defaults.")
            st.session_state.auto_configured = False

# ─────────────────────────────────────────────────────────
# STEP 1 — Understand the dataset
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 1 — Data Understanding", "📊")

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
# STEP 1.5 — Problem Type Identification (Binary vs Multi-Class)
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 1.5 — Target Analysis", "🎯")

st.markdown("""
Identify the **Target Variable** to determine if this is a Binary Classification 
or Multi-Class (3-4+ types) problem. This helps in handling imbalanced data later.
""")

# Heuristic: Identify potential targets (usually object types or low-cardinality numerics)
potential_targets = df.select_dtypes(include=['object']).columns.tolist()
low_card_numeric = [c for c in df.select_dtypes(include='number').columns if df[c].nunique() <= 15]
potential_targets = list(set(potential_targets + low_card_numeric))

# Default selection: Last column (common convention) or first potential target
default_selection = df.columns[-1] 
if default_selection not in potential_targets and len(potential_targets) > 0:
    default_selection = potential_targets[0]

# Selectbox for Target Selection
target_col = st.selectbox(
    "👉 Select your **Target Variable** (Output Column):",
    options=df.columns.tolist(),
    index=list(df.columns).index(default_selection) if default_selection in df.columns else 0,
    key="target_column_selector"
)

if target_col:
    unique_classes = df[target_col].nunique()
    class_counts = df[target_col].value_counts()
    
    # Determine Problem Type & Badge Color
    if unique_classes == 2:
        problem_type = "Binary Classification"
        badge_color = "badge-green"  # Green for simple
    elif unique_classes > 2:
        problem_type = f"Multi-Class Classification ({unique_classes} Classes)"
        badge_color = "badge-orange" # Orange for complex
    else:
        problem_type = "Regression or Single-Class"
        badge_color = "badge-orange"

    # Display Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Target Column", target_col)
    c2.metric("Unique Classes", f"{unique_classes}")
    
    # HTML Badge
    c3.markdown(f"""
    <div style="text-align:center; padding-top:12px;">
        <span class="{badge_color}">{problem_type}</span>
    </div>
    """, unsafe_allow_html=True)

    # Visualization: Class Distribution Bar Chart
    st.markdown("#### 📊 Class Distribution")
    fig, ax = plt.subplots(figsize=(max(8, unique_classes), 5))
    
    # Create barplot
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax, palette="viridis")
    
    # Annotate bars with counts
    for i, v in enumerate(class_counts.values):
        ax.text(i, v + (max(class_counts.values)*0.01), str(v), ha='center', fontsize=9, color='#e2e8f0')

    ax.set_title(f'Distribution of Target: {target_col}', fontsize=12, color='#e2e8f0')
    ax.set_xlabel('Classes', color='#718096')
    ax.set_ylabel('Count', color='#718096')
    ax.tick_params(axis='x', rotation=45 if unique_classes > 5 else 0, colors='#718096')
    ax.tick_params(axis='y', colors='#718096')
    
    # Set dark background
    fig.patch.set_facecolor("#0d1321")
    ax.set_facecolor("#13161f")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2335")
        
    plt.tight_layout()
    st.pyplot(fig)

    # Store target in session state for later steps (like Splitting)
    st.session_state['selected_target'] = target_col
    st.session_state['problem_type'] = problem_type

    # AI Insight
    ai_step_1_5 = ask_deepseek(
        "You are a data scientist. Analyze the class distribution of the target variable. "
        "1) Is the dataset balanced or imbalanced? "
        "2) If imbalanced, suggest 2 techniques to handle it (e.g., SMOTE, Class weights). "
        "3) Confirm if this is a Binary or Multi-class problem.",
        f"Target: {target_col}, Unique Classes: {unique_classes}, Counts: {class_counts.to_dict()}"
    )
    st.markdown(f'<div class="ai-label">🤖 AI ANALYSIS</div><div class="ai-insight">{ai_step_1_5}</div>', unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────
# STEP 2 — Missing values
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 2 — Data Cleaning: Missing Values", "🩹")

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
# STEP 3 — Univariate Analysis
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 3 — Univariate Analysis", "📊")

st.markdown("#### Analyzing Individual Features")

# Separate Features
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

# Identify discrete vs continuous numeric features
discrete_cols = [col for col in num_cols if df[col].nunique() < 20]
continuous_cols = [col for col in num_cols if df[col].nunique() >= 20]

st.markdown(f"**Numeric Features:** {len(num_cols)} total ({len(discrete_cols)} discrete, {len(continuous_cols)} continuous)")
st.markdown(f"**Categorical Features:** {len(cat_cols)}")

# ➤ Continuous Features
if len(continuous_cols) > 0:
    st.markdown("### 📈 Continuous Features")
    st.markdown("**Analysis:** Distribution & Skewness")
    
    # Statistics
    stats_data = []
    for col in continuous_cols[:10]:
        skew = df[col].skew()
        kurt = df[col].kurtosis()
        stats_data.append({
            "Feature": col,
            "Mean": f"{df[col].mean():.2f}",
            "Std": f"{df[col].std():.2f}",
            "Skewness": f"{skew:.2f}",
            "Kurtosis": f"{kurt:.2f}",
            "Distribution": "Normal" if abs(skew) < 0.5 else ("Right-skewed" if skew > 0 else "Left-skewed")
        })
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

    # Visualizations: Histogram, KDE, Box Plot
    st.markdown("**Visualizations:** Histogram & KDE (Distribution) | Box Plot (Outliers)")
    
    n_cols = 3
    n_rows = (len(continuous_cols[:6]) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
    if n_rows == 1: axes = [axes] if n_cols == 1 else axes
    else: axes = axes.flatten()
    
    for idx, col in enumerate(continuous_cols[:6]):
        ax = axes[idx]
        
        # 1. Histogram & KDE
        sns.histplot(df[col], kde=True, ax=ax, color='skyblue', edgecolor='black', alpha=0.6)
        ax.set_title(f'{col}\nSkew: {df[col].skew():.2f}', fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        
        # 2. Box Plot (Inset or Secondary axis could be complex, sticking to simple side-by-side or combined logic if desired. 
        # For clarity in univariate, we will do Box Plots separately below or combined if requested. 
        # Here is a separate Boxplot loop for clarity as requested.)
    
    # Clean up unused axes
    for idx in range(len(continuous_cols[:6]), len(axes)):
        fig.delaxes(axes[idx])
    plt.tight_layout()
    st.pyplot(fig)
    
    # Separate Box Plots for Outlier Detection (Step 3 requirement)
    st.markdown("**Box Plots (Outlier Detection):**")
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_rows == 1: axes2 = [axes2] if n_cols == 1 else axes2
    else: axes2 = axes2.flatten()

    for idx, col in enumerate(continuous_cols[:6]):
        sns.boxplot(x=df[col], ax=axes2[idx], color='lightcoral')
        axes2[idx].set_title(col, fontsize=10)
    
    for idx in range(len(continuous_cols[:6]), len(axes2)):
        fig2.delaxes(axes2[idx])
    plt.tight_layout()
    st.pyplot(fig2)

# ➤ Discrete Features
if len(discrete_cols) > 0:
    st.markdown("### 🔢 Discrete Features")
    st.markdown("**Analysis:** Counts and Spread")
    
    discrete_stats = []
    for col in discrete_cols[:10]:
        discrete_stats.append({
            "Feature": col,
            "Unique Values": df[col].nunique(),
            "Most Common": df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A",
            "Frequency": df[col].value_counts().iloc[0]
        })
    st.dataframe(pd.DataFrame(discrete_stats), use_container_width=True)
    
    # Visualizations: Bar Plot & Box Plot
    st.markdown("**Visualizations:** Bar Plot (Counts) | Box Plot (Spread)")
    
    n_cols = 2
    n_rows = (len(discrete_cols[:4]) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
    if n_rows == 1: axes = [axes] if n_cols == 1 else axes
    else: axes = axes.flatten()
    
    for idx, col in enumerate(discrete_cols[:4]):
        ax = axes[idx]
        df[col].value_counts().sort_index().plot(kind='bar', ax=ax, color='teal', alpha=0.7)
        ax.set_title(f'Count Plot: {col}', fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        
    for idx in range(len(discrete_cols[:4]), len(axes)):
        fig.delaxes(axes[idx])
    plt.tight_layout()
    st.pyplot(fig)

# ➤ Categorical (Object) Features
if len(cat_cols) > 0:
    st.markdown("### 🔤 Categorical Features")
    st.markdown("**Analysis:** Frequency Distribution | Rare Categories Identification")
    
    cat_stats = []
    for col in cat_cols[:10]:
        val_counts = df[col].value_counts(normalize=True)
        rare_count = (val_counts < 0.05).sum() # Count categories with less than 5% frequency
        
        cat_stats.append({
            "Feature": col,
            "Unique Values": df[col].nunique(),
            "Most Common": df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A",
            "Rare Categories (<5%)": rare_count,
            "Type": "High Cardinality" if df[col].nunique() > 10 else "Low Cardinality"
        })
    st.dataframe(pd.DataFrame(cat_stats), use_container_width=True)
    
    # Visualizations: Count Plot
    st.markdown("**Visualizations:** Count Plots")
    
    n_cols = 2
    n_rows = (min(4, len(cat_cols)) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
    if n_rows == 1: axes = [axes] if n_cols == 1 else axes
    else: axes = axes.flatten()
    
    for idx, col in enumerate(cat_cols[:4]):
        ax = axes[idx]
        order = df[col].value_counts().index
        sns.countplot(x=col, data=df, ax=ax, order=order, palette='viridis')
        ax.set_title(f'Count Plot: {col}', fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
    
    for idx in range(len(cat_cols[:4]), len(axes)):
        fig.delaxes(axes[idx])
    plt.tight_layout()
    st.pyplot(fig)


# ─────────────────────────────────────────────────────────
# STEP 4 — Bivariate / Multivariate Analysis
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 4 — Bivariate & Multivariate Analysis", "🔗")

st.markdown("#### Analyzing Relationships Between Features")

# 🔢 Numerical vs Numerical
if len(num_cols) > 1:
    st.markdown("### 🔢 Numerical vs Numerical")
    st.markdown("**Analysis:** Relationship between continuous variables")
    
    # 1. Correlation Heatmap
    st.markdown("**Visualizations:** Correlation Heatmap")
    corr_matrix = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    
    # 2. Scatter Plot & Pairplot
    st.markdown("**Visualizations:** Scatter Plot & Pairplot")
    
    # Select top correlated pairs for scatter plots
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], abs(corr_matrix.iloc[i, j])))
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = corr_pairs[:3]
    
    if top_pairs:
        fig, axes = plt.subplots(1, len(top_pairs), figsize=(15, 5))
        if len(top_pairs) == 1: axes = [axes]
        
        for idx, (col1, col2, corr_val) in enumerate(top_pairs):
            sns.scatterplot(x=df[col1], y=df[col2], ax=axes[idx], alpha=0.6, color='purple')
            axes[idx].set_title(f'{col1} vs {col2} (r={corr_val:.2f})')
        plt.tight_layout()
        st.pyplot(fig)

    # Pairplot (Sampled if data is large to avoid crash)
    if len(num_cols) <= 6: # Only do pairplot for reasonable number of features
        st.markdown("**Pairplot** (Showing relationships and distributions):")
        # Subsample data for pairplot performance
        plot_df = df[num_cols]
        if len(plot_df) > 1000:
            plot_df = plot_df.sample(1000, random_state=42)
            
        fig = sns.pairplot(plot_df, corner=True, plot_kws={'alpha':0.6, 's': 30, 'edgecolor': 'k'})
        fig.fig.suptitle('Pairplot of Numeric Features', y=1.02)
        st.pyplot(fig)

# 🔢 Numerical vs Categorical
if len(cat_cols) > 0 and len(num_cols) > 0:
    st.markdown("### 🔢 Numerical vs Categorical")
    st.markdown("**Analysis:** Compare distributions across categories")
    
    # Select a few combos for visualization
    target_cat = cat_cols[0]
    target_num = num_cols[0]
    
    st.markdown(f"**Visualizations:** Box Plot & Violin Plot ({target_num} by {target_cat})")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box Plot
    sns.boxplot(x=df[target_cat], y=df[target_num], ax=axes[0], palette='Set2')
    axes[0].set_title(f'Box Plot: {target_num} by {target_cat}')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Violin Plot
    # Limit categories if too many for violin plot
    plot_df = df.copy()
    if plot_df[target_cat].nunique() > 15:
        top_cats = plot_df[target_cat].value_counts().index[:10]
        plot_df = plot_df[plot_df[target_cat].isin(top_cats)]
        st.caption(f"Showing top 10 categories for {target_cat} in Violin Plot")

    sns.violinplot(x=plot_df[target_cat], y=plot_df[target_num], ax=axes[1], palette='muted', inner='quartile')
    axes[1].set_title(f'Violin Plot: {target_num} by {target_cat}')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)

# 🔤 Categorical vs Categorical
if len(cat_cols) > 1:
    st.markdown("### 🔤 Categorical vs Categorical")
    st.markdown("**Analysis:** Analyze relationships between categories")
    
    cat1, cat2 = cat_cols[0], cat_cols[1]
    
    st.markdown(f"**Visualizations:** Grouped Bar Chart & Crosstab Heatmap ({cat1} vs {cat2})")
    
    # Grouped Bar Chart (Crosstab normalized usually, but count is safer for general)
    crosstab = pd.crosstab(df[cat1], df[cat2])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Stacked Bar Chart
    crosstab.plot(kind='bar', stacked=True, ax=axes[0], colormap='Paired')
    axes[0].set_title(f'Stacked Bar: {cat1} vs {cat2}')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend(title=cat2, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Crosstab Heatmap
    sns.heatmap(crosstab, annot=True, fmt="d", cmap="Blues", ax=axes[1], linewidths=.5)
    axes[1].set_title(f'Crosstab Heatmap: {cat1} vs {cat2}')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    st.pyplot(fig)

# ⚠️ Outlier Handling (Within Analysis)
st.markdown("### ⚠️ Outlier Handling")
st.markdown("**Method:** IQR (Interquartile Range) | **Decide:** Remove, Cap, Keep")

# Using Box Plots to identify outliers (Already shown in Step 3, but reiterating context here)
num_cols_now = df.select_dtypes(include="number").columns.tolist()
outlier_log = {}
rows_before = len(df)

# Define IQR logic
iqr_factor = 1.5 

# Apply Outlier Handling based on selection or default logic (Here assuming Cap/Remove based on severity)
# For this demo, we will run the detection and propose action

st.markdown("**Outlier Detection & Treatment Log:**")

for col in num_cols_now:
    col_data = df[col].dropna()
    if len(col_data) == 0: continue
        
    Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - iqr_factor * IQR, Q3 + iqr_factor * IQR
    
    n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
    
    if n_out > 0:
        # Decision Logic: If > 5% outliers, Cap. Else Remove (or simple example logic)
        pct_out = n_out / len(df)
        
        if pct_out > 0.05:
            action = "Cap (Winsorization)"
            df[col] = df[col].clip(lower=lo, upper=hi)
            outlier_log[col] = {"outliers": n_out, "action": action, "details": f"Clipped at [{lo:.2f}, {hi:.2f}]"}
        else:
            action = "Remove"
            df = df[(df[col] >= lo) & (df[col] <= hi)]
            outlier_log[col] = {"outliers": n_out, "action": action, "details": f"Bounds [{lo:.2f}, {hi:.2f}]"}

rows_after = len(df)
df.reset_index(drop=True, inplace=True)

col1, col2, col3 = st.columns(3)
col1.metric("Columns with outliers", len(outlier_log))
col2.metric("Rows before", f"{rows_before:,}")
col3.metric("Rows after", f"{rows_after:,}")

if outlier_log:
    out_df = pd.DataFrame([
        {"Column": k, "Outliers": v["outliers"], "Action": v["action"], "Details": v["details"]}
        for k, v in outlier_log.items()
    ])
    st.dataframe(out_df, use_container_width=True)
else:
    st.success("✅ No significant outliers detected requiring action.")

ai_step4 = ask_deepseek(
    "You are a data scientist. Analyze the bivariate relationships and outlier handling: "
    "1) Multicollinearity concerns (Correlation Heatmap) "
    "2) Categorical-numeric patterns (Box/Violin plots) "
    "3) Impact of outlier removal on data distribution. Keep it concise.",
    f"Features analyzed: {len(num_cols)} numeric, {len(cat_cols)} categorical. Outlier actions taken: {len(outlier_log)}.",
)
st.markdown(f'<div class="ai-label">🤖 AI ANALYSIS</div><div class="ai-insight">{ai_step4}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# STEP 5 — Feature Transformation
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 5 — Feature Transformation", "⚙️")

st.markdown("#### Preparing data for Machine Learning")

# 🔢 Numerical Features: Skewness & Scaling
st.markdown("### 🔢 Numerical Features Transformation")
st.markdown("**Goal:** Handle skewness & Scale features")

# Identify skewed features for transformation
skewed_feats = df.select_dtypes(include="number").apply(lambda x: abs(x.skew())).sort_values(ascending=False)
high_skew = skewed_feats[skewed_feats > 0.75].index.tolist()

st.markdown(f"Detected **{len(high_skew)}** highly skewed features (|Skew| > 0.75).")

if len(high_skew) > 0:
    st.markdown("**Visualizations:** Before vs After Distribution (Log/Power Transform)")
    
    # Transform loop for visualization
    n_cols = 2
    n_rows = (min(3, len(high_skew)) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
    if n_rows == 1: axes = [axes] if n_cols == 1 else axes
    else: axes = axes.flatten()
    
    transformed_cols = {}
    
    for idx, col in enumerate(high_skew[:3]):
        ax = axes[idx]
        
        # Plot original
        sns.histplot(df[col], kde=True, ax=ax, color='skyblue', label='Original', stat="density", alpha=0.5)
        
        # Apply Log1p (Log(x+1)) to handle zeros safely
        # Note: In a real pipeline, you'd create new columns. Here we visualize.
        transformed_data = np.log1p(df[col])
        transformed_cols[col] = transformed_data # Store for potential later use
        
        sns.histplot(transformed_data, kde=True, ax=ax, color='purple', label='Log Transformed', stat="density", alpha=0.5)
        
        ax.set_title(f'{col} (Skew: {df[col].skew():.2f} -> {transformed_data.skew():.2f})')
        ax.legend()
        
    for idx in range(len(high_skew[:3]), len(axes)):
        fig.delaxes(axes[idx])
        
    plt.tight_layout()
    st.pyplot(fig)
    
    # Apply transformations to dataframe for the next steps (Scaling)
    for col in high_skew:
        # Ensure data is positive for log transform, otherwise Box-Cox is needed, sticking to Log1p for simplicity
        df[f'{col}_log'] = np.log1p(df[col])

# Scaling
st.markdown("**Scaling:** StandardScaler vs MinMaxScaler")
st.markdown("Applying **StandardScaler** (Mean=0, Std=1) to numerical features (including transformed).")

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scale_cols = df.select_dtypes(include="number").columns.tolist()
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])

# Visualization: Box Plots After Scaling (Shows all features on comparable scale -5 to 5 typically)
st.markdown("**Visualizations:** Box Plots After Scaling (StandardScaler)")
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df_scaled[scale_cols], ax=ax, orient='h', palette='Set3')
ax.set_title('Scaled Numeric Features (StandardScaler)')
ax.set_xlim(-5, 5) # Limit view to see main distribution clearly
st.pyplot(fig)

# 🔤 Categorical Features (Encoding)
st.markdown("### 🔤 Categorical Features (Encoding Decision)")
st.markdown("**Strategy:** Label Encoding (Ordinal) vs One-Hot (Nominal)")

cat_cols_now = df.select_dtypes(include="object").columns.tolist()
encoding_map = {}

for col in cat_cols_now:
    unique_count = df[col].nunique()
    
    if unique_count == 2:
        # Binary: Label Encoding
        encoding_map[col] = "Label Encoding (Binary)"
        # Apply simple map
        mapping = {k: i for i, k in enumerate(df[col].unique())}
        df[col] = df[col].map(mapping)
        
    elif unique_count <= 10:
        # Low Cardinality: One-Hot Encoding
        encoding_map[col] = "One-Hot Encoding (Low Cardinality)"
        # Apply One-Hot
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        
    else:
        # High Cardinality: Target/Frequency Encoding (Using Frequency here for safety without target)
        encoding_map[col] = "Frequency Encoding (High Cardinality)"
        freq = df[col].value_counts(normalize=True)
        df[col + '_freq'] = df[col].map(freq)
        df.drop(columns=[col], inplace=True)

# Display Encoding Decisions
enc_df = pd.DataFrame([
    {"Column": k, "Strategy": v} for k, v in encoding_map.items()
])
st.dataframe(enc_df, use_container_width=True)

st.markdown("**Visualizations:** Category Distribution Before Encoding (Representative)")
if len(cat_cols_now) > 0:
    # Pick one original categorical to show distribution
    # Since we modified df, we ideally need original, but assuming context for visualization
    st.caption("Encoding effectively expands feature space (One-Hot) or compresses info (Target/Freq).")

ai_step5 = ask_deepseek(
    "You are a data scientist. Explain the feature transformation steps: "
    "1) Why Log transform helps skewness "
    "2) Difference between StandardScaler and MinMaxScaler and when to use which "
    "3) Encoding strategies (Ordinal vs Nominal vs High Cardinality). Keep it to 5 bullet points.",
    f"Skewed features handled: {len(high_skew)}. Encoding strategies applied: {len(encoding_map)}.",
)
st.markdown(f'<div class="ai-label">🤖 AI ANALYSIS</div><div class="ai-insight">{ai_step5}</div>', unsafe_allow_html=True)

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
# In Step 9...
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 9 — Train / Test Split", "✂️")

target_variable = st.session_state.get('selected_target', None)

n_total = len(df)
n_test = int(n_total * test_size)
n_train = n_total - n_test

# Check if we can use Stratified Split (Requires a target variable with sufficient samples)
if target_variable and target_variable in df.columns:
    # Ensure each class has at least 2 samples to avoid errors in stratify
    class_counts = df[target_variable].value_counts()
    min_class_count = class_counts.min()
    
    if min_class_count >= 2:
        st.info(f"🎯 Using **Stratified Split** on target `{target_variable}` to maintain class ratios.")
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=42, 
            stratify=df[target_variable]
        )
    else:
        st.warning(f"⚠️ Target `{target_variable}` has a class with only {min_class_count} sample(s). Using random split instead.")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
else:
    st.info("ℹ️ No target variable selected. Using random split.")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

# ... Rest of your Step 9 metrics and visualizations ...
col1, col2, col3 = st.columns(3)
col1.metric("Total samples", f"{n_total:,}")
col2.metric("Train set", f"{len(train_df):,}  ({100*(1-test_size):.0f}%)")
col3.metric("Test set", f"{len(test_df):,}  ({100*test_size:.0f}%)")

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

# 1.1 Drop ID-like columns (>95% unique values)
id_cols = [col for col in df.columns if df[col].nunique() / len(df) > 0.95]
if id_cols:
    df.drop(columns=id_cols, inplace=True)
    print(f"Dropped ID columns: {{id_cols}}")

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

# ─────────────────────────────────────────────────────────
# COMPREHENSIVE AI ANALYSIS & INSIGHTS
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🎯 Comprehensive Data Analysis & Insights")

# Collect all preprocessing statistics
num_cols_original = raw_df.select_dtypes(include="number").columns.tolist()
cat_cols_original = raw_df.select_dtypes(include="object").columns.tolist()
missing_before = raw_df.isnull().sum().sum()
missing_after = df.isnull().sum().sum()
rows_removed = raw_df.shape[0] - df.shape[0]
cols_removed = raw_df.shape[1] - df.shape[1]

# ─────────────────────────────────────────────────────────
# EXPLORATORY DATA ANALYSIS VISUALIZATIONS
# ─────────────────────────────────────────────────────────
st.markdown("### 📊 Exploratory Data Analysis")

# 1. Distribution Analysis for Numeric Features
if len(num_cols_original) > 0:
    st.markdown("#### Distribution of Numeric Features")
    n_cols = min(3, len(num_cols_original))
    n_rows = (len(num_cols_original) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(num_cols_original[:min(9, len(num_cols_original))]):
        ax = axes[idx] if len(num_cols_original) > 1 else axes[0]
        raw_df[col].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f'{col} Distribution')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    
    for idx in range(len(num_cols_original), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    st.pyplot(fig)

# 2. Correlation Heatmap
if len(num_cols_original) > 1:
    st.markdown("#### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = raw_df[num_cols_original].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    st.pyplot(fig)

# 3. Box Plots for Outlier Detection
if len(num_cols_original) > 0:
    st.markdown("#### Box Plots - Outlier Detection")
    n_cols = min(3, len(num_cols_original))
    n_rows = (len(num_cols_original) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(num_cols_original[:min(9, len(num_cols_original))]):
        ax = axes[idx] if len(num_cols_original) > 1 else axes[0]
        raw_df.boxplot(column=col, ax=ax)
        ax.set_title(f'{col} - Outliers')
        ax.set_ylabel(col)
    
    for idx in range(len(num_cols_original), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    st.pyplot(fig)

# 4. Categorical Feature Analysis
if len(cat_cols_original) > 0:
    st.markdown("#### Categorical Features Distribution")
    n_cols = min(2, len(cat_cols_original))
    n_rows = (len(cat_cols_original) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(cat_cols_original[:min(6, len(cat_cols_original))]):
        ax = axes[idx] if len(cat_cols_original) > 1 else axes[0]
        value_counts = raw_df[col].value_counts().head(10)
        value_counts.plot(kind='bar', ax=ax, color='coral')
        ax.set_title(f'{col} Distribution')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
    
    for idx in range(len(cat_cols_original), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    st.pyplot(fig)

# 5. Group-by Analysis (if categorical columns exist)
if len(cat_cols_original) > 0 and len(num_cols_original) > 0:
    st.markdown("#### Group-by Analysis - Categorical vs Numeric")
    
    # Select first categorical and numeric column for demo
    cat_col = cat_cols_original[0]
    num_col = num_cols_original[0]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    grouped_data = raw_df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(10)
    grouped_data.plot(kind='bar', ax=ax, color='teal')
    ax.set_title(f'Average {num_col} by {cat_col}')
    ax.set_xlabel(cat_col)
    ax.set_ylabel(f'Average {num_col}')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# Generate comprehensive analysis prompt
analysis_data = f"""
ORIGINAL DATASET:
- Shape: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns
- Numeric columns ({len(num_cols_original)}): {num_cols_original[:5]}
- Categorical columns ({len(cat_cols_original)}): {cat_cols_original[:5]}
- Missing values: {missing_before}
- Data types distribution: {dict(raw_df.dtypes.value_counts())}

PREPROCESSING APPLIED:
1. Missing Values: {missing_strategy} → Removed {missing_before - missing_after} missing values
2. Duplicates: Removed {raw_df.duplicated().sum()} duplicate rows
3. Outliers: {outlier_method} method with {outlier_action} action (threshold: {iqr_factor if outlier_method == 'IQR' else z_threshold})
4. Encoding: {encoding_method} (max unique for one-hot: {max_onehot_unique})
5. Scaling: {scaling_method}
6. Feature Selection: Removed {cols_removed} low-variance/highly-correlated features
7. Train/Test Split: {int((1-test_size)*100)}% train / {int(test_size*100)}% test

FINAL DATASET:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Rows removed: {rows_removed} ({(rows_removed/raw_df.shape[0]*100):.1f}%)
- Columns removed: {cols_removed}
- Missing values: {missing_after}

CORRELATION INSIGHTS:
{corr_matrix.unstack().sort_values(ascending=False).head(10).to_dict() if len(num_cols_original) > 1 else 'N/A'}

GROUP-BY INSIGHTS:
{grouped_data.to_dict() if len(cat_cols_original) > 0 and len(num_cols_original) > 0 else 'N/A'}
"""

ai_comprehensive = ask_deepseek(
    """You are a senior data scientist. Analyze the EDA results and provide insights.

### PHASE 1: EXPLORATORY DATA ANALYSIS

**Step 1 - Data Understanding:**
- Dataset structure and basic statistics
- Data types appropriateness

**Step 2 - Data Cleaning:**
- Missing values handled
- Duplicates removed
- ID columns dropped

**Step 3 - Univariate Analysis:**
- Distribution patterns (histograms)
- Outliers (box plots)

**Step 4 - Bivariate Analysis:**
- Correlations from heatmap
- Group-by insights

**Step 5 - Visualization Summary:**
- Key patterns discovered

### PHASE 2: FEATURE ENGINEERING (For ML)

**Step 6 - Transformation:**
- Encoding: {encoding_method}
- Scaling: {scaling_method}

**Step 7 - Selection:**
- Features removed: {cols_removed}

**Step 8 - Split:**
- Train/Test: {int((1-test_size)*100)}/{int(test_size*100)}

### RECOMMENDATIONS:
- ML models suitable for this data
- Next steps

Be concise and actionable.""",
    analysis_data,
    max_tokens=2000,
)

st.markdown(f'<div class="ai-insight">{ai_comprehensive}</div>', unsafe_allow_html=True)

# Summary metrics
st.markdown("### 📈 Pipeline Summary")
summary_cols = st.columns(5)
summary_cols[0].metric("Data Retained", f"{(df.shape[0]/raw_df.shape[0]*100):.1f}%")
summary_cols[1].metric("Features", f"{df.shape[1]}", delta=f"{cols_removed}" if cols_removed != 0 else "0")
summary_cols[2].metric("Missing Values", f"{missing_after}", delta=f"-{missing_before - missing_after}")
summary_cols[3].metric("Duplicates Removed", f"{raw_df.duplicated().sum()}")
summary_cols[4].metric("Train/Test", f"{int((1-test_size)*100)}/{int(test_size*100)}")
