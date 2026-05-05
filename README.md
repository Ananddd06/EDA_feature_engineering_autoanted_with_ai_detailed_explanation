# ML Preprocessing Agent

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#installation)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b.svg)](#run-the-app)
[![OpenRouter](https://img.shields.io/badge/AI-OpenRouter-6f42c1.svg)](#detailed-implementation-openrouter-ai-explanations)

Interactive Streamlit app for end-to-end tabular data preprocessing with optional AI explanations powered by OpenRouter.

Upload a CSV and get:

- full preprocessing pipeline execution
- visual diagnostics at each stage
- downloadable cleaned dataset
- downloadable reproducible Python script
- AI commentary for interview-style understanding

---

## Table of contents

- [Key features](#key-features)
- [Project structure](#project-structure)
- [Tech stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Run the app](#run-the-app)
- [Pipeline walkthrough](#pipeline-walkthrough)
- [Detailed implementation: OpenRouter AI explanations](#detailed-implementation-openrouter-ai-explanations)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [Development notes](#development-notes)
- [License](#license)

---

## Key features

- CSV-driven workflow for quick preprocessing
- Smart missing-value handling options
- Duplicate detection/removal
- Univariate + multivariate visual analysis
- Outlier treatment (IQR / Z-score + cap/remove logic)
- Categorical encoding strategies
- Model-training-aware preprocessing guidance (scaling deferred to training)
- Feature engineering and correlated-feature removal
- Target-aware split with stratification fallback where possible
- End-to-end post-cleaning model training, evaluation, cross-validation, and tuning
- User-adjustable model validation controls (CV folds and tuning iterations)
- OpenRouter-based AI explanations and auto-configuration support

---

## Project structure

```text
preprocessing_agent/
├── preprocessing_agent.py   # Main app (recommended entrypoint)
├── app.py                   # Alternate pipeline variant
├── requirements.txt         # Python dependencies
├── EDA_Visa_US.ipynb        # Notebook artifact
└── README.md
```

---

## Tech stack

- Python 3.9+
- Streamlit
- pandas, numpy
- scikit-learn
- scipy
- matplotlib, seaborn
- python-dotenv
- OpenAI Python SDK (used against OpenRouter base URL)

---

## Installation

```bash
git clone <your-repo-url>
cd preprocessing_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Configuration

AI explanations are optional. Core preprocessing runs locally regardless.

Create `.env` in project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

If `OPENROUTER_API_KEY` is missing, AI sections return a friendly warning string and the app continues normally.

---

## Run the app

Recommended:

```bash
streamlit run preprocessing_agent.py
```

> `preprocessing_agent.py` includes the full pipeline with end-to-end model training.

Alternate:

```bash
streamlit run app.py
```

> `app.py` is a preprocessing-focused variant and does not include the model training stage.

Open the URL shown by Streamlit (typically `http://localhost:8501`).

---

## Pipeline walkthrough

The primary app (`preprocessing_agent.py`) executes a rich preprocessing + EDA flow:

1. **Data understanding**  
   Shape, types, null counts, descriptive stats, and AI summary.

2. **Target analysis (Step 1.5)**  
   User selects target column; app shows class counts and problem-type hint (binary/multi-class/regression-like).

3. **Missing value handling**  
   Configurable strategy: auto/mean/median/mode/drop rows.

4. **Duplicate removal**  
   Detects and removes exact row duplicates.

5. **Univariate analysis**  
   Histograms/distributions/skewness style insights.

6. **Bivariate & multivariate analysis**  
   Correlation heatmap, scatter/pair plots, categorical interactions.

7. **Feature transformation**  
   Skew handling (log-style transformations) and categorical encoding strategy.

8. **Feature engineering**  
   Correlation diagnostics and zero-variance feature checks.

9. **Feature selection**  
   Drops highly correlated features above threshold.

10. **Train/test split + export**  
     Stratified split when target allows; otherwise random split. Exports cleaned CSV + generated Python script.

11. **Post-cleaning model development**  
    Automatically detects classification/regression, runs multi-model training, CV, hyperparameter tuning, and final test-set evaluation with diagnostics.

---

## Detailed implementation: OpenRouter AI explanations

This section describes exactly how your AI explanation layer is implemented.

### 1) Environment + client bootstrapping

In `preprocessing_agent.py`, you load env vars and initialize an OpenAI-compatible client pointing to OpenRouter:

```python
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
client = OpenAI(
    api_key=OPENROUTER_API_KEY or "placeholder",
    base_url=OPENROUTER_BASE_URL,
)
AI_ENABLED = bool(OPENROUTER_API_KEY)
```

Key idea: same OpenAI SDK, but `base_url` routes requests to OpenRouter.

### 2) Central AI wrapper (`ask_deepseek`)

All AI calls go through one helper:

```python
def ask_deepseek(system: str, user: str, max_tokens: int = 1500) -> str:
    if not AI_ENABLED:
        return "⚠️ Set `OPENROUTER_API_KEY` in your environment to enable AI commentary."
    try:
        resp = client.chat.completions.create(
            model="z-ai/glm-4.5-air:free",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"⚠️ OpenRouter error: {exc}"
```

Design benefits:

- one model/config control point
- graceful fallback when API key is missing
- explicit error surfacing in UI if request fails

### 3) Prompt strategy across the pipeline

You use task-specific prompts at each major step:

- step summaries (data understanding, missing values, deduplication, etc.)
- interview-focused bullet-point explanations
- concise, role-driven system prompts (e.g., “You are a senior ML engineer…”)

You pass structured context (`pretty_json(...)`, counts, shapes, config choices) so responses remain grounded in actual dataset state.

### 4) AI auto-configuration flow

`preprocessing_agent.py` includes an AI-driven parameter recommendation phase that:

- builds a compact dataset summary (`shape`, type counts, missing stats, unique counts)
- requests a strict JSON response with required keys:
  - `missing_strategy`
  - `outlier_method`
  - `outlier_action`
  - `z_threshold`
  - `iqr_factor`
  - `encoding_method`
  - `test_size`
  - `max_onehot_unique`
  - `reasoning`
- parses and writes values into `st.session_state`
- reruns app with updated controls

This gives users a guided “AI suggested config” before running the rest of the pipeline.

### 5) Runtime fail-safe behavior

Your implementation is robust in degraded AI conditions:

- no API key → preprocessing still works, AI message is informative
- API call exception → error string returned, app remains interactive
- JSON parse failure during auto-config → warning shown and defaults preserved

So AI is an enhancement layer, not a hard dependency.

### 6) Why this architecture is good

- clean separation: deterministic preprocessing vs narrative AI explanation
- minimal coupling: one helper function mediates all model access
- easy model swap: change only `model="..."` in one place
- reproducible data processing remains local and transparent

---

## Outputs

After each run, the app can provide:

- transformed dataset preview
- cleaned CSV download
- generated Python preprocessing script download
- per-step visualizations (distributions, correlations, class plots, etc.)
- AI commentary blocks and final summary

---

## Troubleshooting

### `streamlit: command not found`

Activate venv and reinstall:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### AI explanation not appearing

- Verify `OPENROUTER_API_KEY`
- Verify network connectivity
- Verify model access/credits on OpenRouter

### Dependency issues

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Development notes

- Primary app is `preprocessing_agent.py`
- Keep heavy preprocessing deterministic (pandas/scikit-learn)
- Keep AI usage explainability-focused and optional
- Update `requirements.txt` when adding/removing dependencies

---

## License

No `LICENSE` file is currently present.  
Add one before public distribution.
