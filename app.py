"""
ML Preprocessing Agent — powered by OpenRouter (z-ai/glm-4.5-air:free)
Run:  streamlit run preprocessing_agent.py
Requires: pip install streamlit pandas scikit-learn matplotlib seaborn scipy python-dotenv openai tabulate
"""

import io
import json
import os
import warnings
import html
import re
import difflib
from collections import Counter
from typing import Optional
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import (
    ParameterSampler,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
)

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    SMOTE = None
    ImbPipeline = None
    IMBLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None
    LIGHTGBM_AVAILABLE = False

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


def safe_numeric_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    return numeric


def safe_categorical_series(series: pd.Series) -> pd.Series:
    return series.map(lambda v: str(v) if pd.notna(v) else np.nan)


def safe_value_counts(series: pd.Series, normalize: bool = False):
    clean = safe_categorical_series(series).dropna()
    if clean.empty:
        return pd.Series(dtype=float if normalize else int)
    counts = Counter(clean.tolist())
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    keys = [k for k, _ in items]
    vals = np.array([v for _, v in items], dtype=float if normalize else int)
    if normalize:
        total = float(vals.sum())
        if total > 0:
            vals = vals / total
    out = pd.Series(vals, index=keys, dtype=float if normalize else int)
    return out


def plot_hist_kde_safe(
    ax,
    series: pd.Series,
    color: str = "skyblue",
    label: Optional[str] = None,
    density: bool = False,
    alpha: float = 0.6,
):
    values = np.asarray(series, dtype=float)
    if values.size == 0:
        return False
    bins = min(40, max(10, int(np.sqrt(values.size))))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return False

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmin == vmax:
        ax.axvline(vmin, color=color, alpha=alpha, linewidth=2, label=label)
        return True

    edges = np.linspace(vmin, vmax, bins + 1)
    bin_idx = np.searchsorted(edges, values, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, bins - 1)
    counts = np.bincount(bin_idx, minlength=bins).astype(float)

    widths = np.diff(edges)
    if density and counts.sum() > 0:
        counts = counts / (counts.sum() * widths)

    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(
        centers,
        counts,
        width=widths * 0.95,
        color=color,
        alpha=alpha,
        edgecolor="black",
        align="center",
        label=label,
    )
    try:
        sns.kdeplot(values, ax=ax, color=color, linewidth=1.6, warn_singular=False)
    except Exception:
        pass
    return True


def parse_json_response(raw_text: str):
    cleaned = (raw_text or "").strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise ValueError("Could not parse JSON response from OpenRouter.")


def _derive_learning_type_from_problem(problem_text: str) -> str:
    text = (problem_text or "").lower()
    unsupervised_hits = any(k in text for k in ["cluster", "segmentation", "segment", "anomaly", "grouping", "pattern"])
    supervised_hits = any(
        k in text
        for k in [
            "predict",
            "prediction",
            "classify",
            "classification",
            "regression",
            "forecast",
            "estimate",
            "target",
            "label",
            "outcome",
        ]
    )
    if unsupervised_hits and not supervised_hits:
        return "unsupervised"
    if supervised_hits:
        return "supervised"
    return "unsupervised"


def _derive_task_type_from_text(problem_text: str) -> str:
    text = (problem_text or "").lower()
    regression_hits = any(
        k in text
        for k in ["regression", "forecast", "estimate", "continuous", "numeric value", "real value"]
    )
    classification_hits = any(
        k in text
        for k in ["class", "label", "churn", "fraud", "default", "approve", "reject", "classification"]
    )
    if regression_hits and not classification_hits:
        return "regression"
    if classification_hits:
        return "classification"
    return "classification"


def _find_target_from_problem_text(df: pd.DataFrame, text: str) -> Optional[str]:
    if df.empty:
        return None
    raw_text = (text or "").lower()
    words = re.findall(r"[a-zA-Z_]+", raw_text)
    word_set = set(words)

    scored = []
    for col in df.columns:
        col_l = col.lower()
        col_tokens = re.findall(r"[a-zA-Z_]+", col_l)
        score = 0.0

        if col_l in raw_text:
            score += 6.0

        token_hits = sum(1 for tok in col_tokens if tok in word_set and len(tok) >= 3)
        score += 2.0 * token_hits

        close = difflib.get_close_matches(col_l, list(word_set), n=1, cutoff=0.75)
        if close:
            score += 3.0

        for tok in col_tokens:
            if len(tok) < 4:
                continue
            if difflib.get_close_matches(tok, list(word_set), n=1, cutoff=0.8):
                score += 1.5

        if score > 0:
            scored.append((score, col))

    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def ai_target_detection(
    df: pd.DataFrame,
    dataset_description: str = "",
    problem_statement: str = "",
    basic_eda_summary: Optional[dict] = None,
):
    col_stats = []
    for col in df.columns:
        series = df[col]
        col_stats.append(
            {
                "name": col,
                "dtype": str(series.dtype),
                "missing_pct": round(float(series.isna().mean() * 100), 3),
                "n_unique": int(series.nunique(dropna=True)),
                "unique_ratio": round(float(series.nunique(dropna=True) / max(len(series), 1)), 4),
            }
        )

    prompt_payload = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": col_stats,
        "dataset_description": dataset_description.strip(),
        "problem_statement": problem_statement.strip(),
        "basic_eda_summary": basic_eda_summary or {},
    }
    ai_raw = ask_deepseek(
        system=(
            "You are a senior ML scientist. Infer whether the dataset is supervised or unsupervised and "
            "recommend the best target column(s) from schema/statistics and user intent. Return ONLY a valid JSON object."
        ),
        user=(
            "Analyze this dataset summary and return JSON with EXACT keys:\n"
            "learning_type, task_type, confidence, recommended_target, alternative_targets, reasoning, cautions.\n"
            "Rules:\n"
            "- learning_type: 'supervised' or 'unsupervised'\n"
            "- task_type: 'classification', 'regression', or 'unsupervised'\n"
            "- confidence: float between 0 and 1\n"
            "- recommended_target: string column name or null\n"
            "- alternative_targets: array of up to 5 column names\n"
            "- reasoning: concise string (2-4 lines)\n"
            "- cautions: concise string\n"
            "Use column names exactly as provided.\n"
            f"Dataset summary:\n{pretty_json(prompt_payload)}"
        ),
        max_tokens=900,
    )
    try:
        parsed = parse_json_response(ai_raw)
    except Exception:
        parsed = {}

    merged_problem_text = f"{problem_statement} {dataset_description}".strip()
    heuristic_text_target = _find_target_from_problem_text(df, merged_problem_text)
    strong_supervised_signal = (
        heuristic_text_target is not None
        or any(
            k in merged_problem_text.lower()
            for k in ["predict", "forecast", "estimate", "classify", "classification", "regression", "target", "label", "outcome"]
        )
    )
    fallback_learning_type = _derive_learning_type_from_problem(merged_problem_text)
    if strong_supervised_signal:
        fallback_learning_type = "supervised"
    learning_type = str(parsed.get("learning_type", fallback_learning_type)).strip().lower()
    if learning_type not in {"supervised", "unsupervised"}:
        learning_type = fallback_learning_type
    if learning_type == "unsupervised" and strong_supervised_signal:
        learning_type = "supervised"

    task_type = str(parsed.get("task_type", "")).strip().lower()
    if task_type not in {"classification", "regression", "unsupervised"}:
        if learning_type == "unsupervised":
            task_type = "unsupervised"
        else:
            task_type = _derive_task_type_from_text(merged_problem_text)

    recommended_target = parsed.get("recommended_target")
    if isinstance(recommended_target, str):
        recommended_target = recommended_target.strip()
    else:
        recommended_target = None
    if recommended_target and recommended_target not in df.columns:
        recommended_target = None

    if recommended_target is None and learning_type == "supervised":
        if heuristic_text_target in df.columns:
            recommended_target = heuristic_text_target

    if recommended_target is None and learning_type == "supervised":
        problem_text = f"{problem_statement} {dataset_description} {ai_raw}".lower()
        for col in df.columns:
            if re.search(rf"\b{re.escape(col.lower())}\b", problem_text):
                recommended_target = col
                break

    if recommended_target is None and learning_type == "supervised":
        candidate_targets = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        candidate_targets += [c for c in df.select_dtypes(include=["number"]).columns if df[c].nunique(dropna=True) <= 20]
        recommended_target = candidate_targets[0] if candidate_targets else None

    if learning_type == "unsupervised":
        recommended_target = None
        task_type = "unsupervised"
    elif recommended_target in df.columns:
        task_type = infer_problem_type(df[recommended_target])
    elif learning_type == "supervised":
        task_type = _derive_task_type_from_text(merged_problem_text)

    alternative_targets = parsed.get("alternative_targets", [])
    if not isinstance(alternative_targets, list):
        alternative_targets = []
    alternative_targets = [str(c).strip() for c in alternative_targets if str(c).strip() in df.columns]

    confidence = parsed.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.55 if parsed else 0.35
    confidence = max(0.0, min(1.0, confidence))

    return {
        "learning_type": learning_type,
        "task_type": task_type,
        "confidence": confidence,
        "recommended_target": recommended_target,
        "alternative_targets": alternative_targets,
        "reasoning": str(parsed.get("reasoning", "")).strip() or (
            "Used dataset schema + problem statement to infer learning mode and target recommendation."
        ),
        "cautions": str(parsed.get("cautions", "")).strip() or (
            "OpenRouter response was overridden by strong supervised signals from your problem statement."
            if learning_type == "supervised" and strong_supervised_signal
            else (
                "OpenRouter response was not strict JSON; fallback parsing/heuristics were applied."
                if not parsed
                else ""
            )
        ),
        "raw_text": ai_raw,
    }


def build_unsupervised_feature_matrix(dataframe: pd.DataFrame):
    X = dataframe.copy()
    X = X.dropna(axis=1, how="all")
    if X.empty:
        return pd.DataFrame(), {}

    datetime_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    for col in datetime_cols:
        X[col] = pd.to_datetime(X[col], errors="coerce").astype("int64") / 1e9

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())
    for col in cat_cols:
        mode_series = X[col].mode(dropna=True)
        fill_value = mode_series.iloc[0] if not mode_series.empty else "missing"
        X[col] = X[col].astype(str).fillna(fill_value)

    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    vt = VarianceThreshold(threshold=0.0)
    try:
        vt.fit(X)
        X = X[X.columns[vt.get_support()]]
    except Exception:
        pass

    summary = {
        "rows": int(X.shape[0]),
        "features_after_encoding": int(X.shape[1]),
        "datetime_cols_converted": datetime_cols,
    }
    return X, summary


def fig_to_buf(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def section(title: str, icon: str = ""):
    icon_html = f"<span class='section-icon'>{icon}</span>" if icon else ""
    st.markdown(
        f"<div class='section-title'>{icon_html}<span>{title}</span></div>",
        unsafe_allow_html=True,
    )


def render_training_status_box(container, logs, status: str = "running"):
    status_title = {
        "running": "⚙️ Model training in progress",
        "complete": "✅ Model training complete",
    }
    safe_logs = "<br>".join(f"• {html.escape(line)}" for line in logs[-12:])
    container.markdown(
        f"""
        <div class="train-box train-{status}">
            <div class="train-box-title">{status_title.get(status, status_title["running"])}</div>
            <div class="train-box-log">{safe_logs}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def detect_primary_id_columns(dataframe: pd.DataFrame):
    """
    Detect only true primary identifier columns (e.g., case_id, customer_id),
    instead of dropping every high-uniqueness feature.
    """
    if dataframe.empty:
        return []

    protected_target_guess = dataframe.columns[-1]  # keep likely supervised target safe
    id_columns = []
    id_name_tokens = ("id", "uuid", "identifier", "case_no", "caseid", "record_no")

    for col in dataframe.columns:
        if col == protected_target_guess:
            continue

        series = dataframe[col].dropna()
        if series.empty:
            continue

        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.995:
            continue

        col_name = col.lower()
        name_suggests_id = any(token in col_name for token in id_name_tokens)

        index_like_sequence = False
        if pd.api.types.is_integer_dtype(series):
            diffs = series.diff().dropna()
            if not diffs.empty:
                step_one_ratio = (diffs.abs() == 1).mean()
                index_like_sequence = step_one_ratio >= 0.98

        if name_suggests_id or index_like_sequence:
            id_columns.append(col)

    return id_columns


def detect_target_column(dataframe: pd.DataFrame):
    if "selected_target" in st.session_state:
        target = st.session_state.get("selected_target")
        if target in dataframe.columns:
            return target
    prioritized_names = {"target", "label", "class", "outcome", "y"}
    for col in dataframe.columns:
        if col.lower() in prioritized_names:
            return col
    return dataframe.columns[-1] if len(dataframe.columns) > 1 else None


def infer_problem_type(target_series: pd.Series):
    if target_series.dtype == "object" or str(target_series.dtype).startswith("category") or target_series.dtype == "bool":
        return "classification"
    n_unique = target_series.nunique(dropna=True)
    n_rows = len(target_series)
    if pd.api.types.is_numeric_dtype(target_series) and n_unique > max(15, int(0.05 * n_rows)):
        return "regression"
    return "classification"


def build_model_pipeline(estimator, scaler_name: str, use_scaler: bool, use_smote: bool):
    scaler = StandardScaler() if scaler_name == "StandardScaler" else MinMaxScaler()
    if use_smote and IMBLEARN_AVAILABLE and ImbPipeline is not None:
        steps = []
        if use_scaler:
            steps.append(("scaler", scaler))
        steps.append(("smote", SMOTE(random_state=42)))
        steps.append(("model", estimator))
        return ImbPipeline(steps=steps)
    steps = []
    if use_scaler:
        steps.append(("scaler", scaler))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)


def get_model_registry(problem_type: str):
    if problem_type == "classification":
        models = {
            "Logistic Regression": {
                "estimator": LogisticRegression(max_iter=2000),
                "use_scaler": True,
                "params": {
                    "model__C": [0.01, 0.1, 1.0, 10.0],
                    "model__solver": ["lbfgs", "liblinear"],
                },
            },
            "K-Nearest Neighbors": {
                "estimator": KNeighborsClassifier(),
                "use_scaler": True,
                "params": {
                    "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
                    "model__weights": ["uniform", "distance"],
                    "model__p": [1, 2],
                },
            },
            "Stochastic Gradient Descent (Classifier)": {
                "estimator": SGDClassifier(
                    loss="log_loss",
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=8,
                    random_state=42,
                ),
                "use_scaler": True,
                "params": {
                    "model__loss": ["log_loss", "modified_huber"],
                    "model__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                    "model__penalty": ["l2", "l1", "elasticnet"],
                    "model__l1_ratio": [0.15, 0.5, 0.85],
                    "model__learning_rate": ["optimal", "adaptive"],
                    "model__eta0": [0.001, 0.01, 0.1],
                    "model__max_iter": [1000, 2000, 3000],
                },
            },
            "Decision Tree": {
                "estimator": DecisionTreeClassifier(random_state=42),
                "use_scaler": False,
                "params": {
                    "model__max_depth": [None, 5, 10, 20],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 2, 4],
                },
            },
            "Random Forest": {
                "estimator": RandomForestClassifier(random_state=42),
                "use_scaler": False,
                "params": {
                    "model__n_estimators": [100, 200, 300],
                    "model__max_depth": [None, 8, 16, 24],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 2, 4],
                },
            },
            "Gradient Boosting": {
                "estimator": GradientBoostingClassifier(random_state=42),
                "use_scaler": False,
                "params": {
                    "model__n_estimators": [100, 200, 300],
                    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "model__max_depth": [2, 3, 4],
                },
            },
            "Naive Bayes": {
                "estimator": GaussianNB(),
                "use_scaler": True,
                "params": {"model__var_smoothing": np.logspace(-11, -7, 9)},
            },
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = {
                "estimator": XGBClassifier(eval_metric="logloss", random_state=42),
                "use_scaler": False,
                "params": {
                    "model__n_estimators": [100, 200, 300],
                    "model__max_depth": [3, 5, 7],
                    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "model__subsample": [0.8, 1.0],
                },
            }
        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = {
                "estimator": LGBMClassifier(random_state=42, verbose=-1),
                "use_scaler": False,
                "params": {
                    "model__n_estimators": [100, 200, 300],
                    "model__num_leaves": [15, 31, 63],
                    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "model__subsample": [0.8, 1.0],
                },
            }
        return models

    models = {
        "Linear Regression": {
            "estimator": LinearRegression(),
            "use_scaler": True,
            "params": {},
        },
        "Ridge": {
            "estimator": Ridge(random_state=42),
            "use_scaler": True,
            "params": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        },
        "Lasso": {
            "estimator": Lasso(random_state=42, max_iter=5000),
            "use_scaler": True,
            "params": {"model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
        },
        "Decision Tree Regressor": {
            "estimator": DecisionTreeRegressor(random_state=42),
            "use_scaler": False,
            "params": {
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        },
        "Random Forest Regressor": {
            "estimator": RandomForestRegressor(random_state=42),
            "use_scaler": False,
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 8, 16, 24],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        },
        "Gradient Boosting Regressor": {
            "estimator": GradientBoostingRegressor(random_state=42),
            "use_scaler": False,
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__max_depth": [2, 3, 4],
            },
        },
        "K-Nearest Neighbors Regressor": {
            "estimator": KNeighborsRegressor(),
            "use_scaler": True,
            "params": {
                "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        },
        "Stochastic Gradient Descent (Regressor)": {
            "estimator": SGDRegressor(
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=8,
                random_state=42,
            ),
            "use_scaler": True,
            "params": {
                "model__loss": ["squared_error", "huber"],
                "model__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                "model__penalty": ["l2", "l1", "elasticnet"],
                "model__l1_ratio": [0.15, 0.5, 0.85],
                "model__learning_rate": ["optimal", "adaptive", "invscaling"],
                "model__eta0": [0.001, 0.01, 0.1],
                "model__max_iter": [1000, 2000, 3000],
            },
        },
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost Regressor"] = {
            "estimator": XGBRegressor(random_state=42),
            "use_scaler": False,
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__subsample": [0.8, 1.0],
            },
        }
    if LIGHTGBM_AVAILABLE:
        models["LightGBM Regressor"] = {
            "estimator": LGBMRegressor(random_state=42, verbose=-1),
            "use_scaler": False,
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__num_leaves": [15, 31, 63],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__subsample": [0.8, 1.0],
            },
        }
    return models


def get_search_space_size(param_dist: dict) -> int:
    if not param_dist:
        return 1
    size = 1
    for values in param_dist.values():
        size *= len(list(values))
    return max(1, size)


def score_to_loss(problem_type: str, score: float) -> float:
    if problem_type == "classification":
        return max(0.0, 1.0 - float(score))
    return float(-score)


def evaluate_predictions(problem_type: str, model, X_eval: pd.DataFrame, y_true: pd.Series):
    y_pred = model.predict(X_eval)
    if problem_type == "classification":
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1-score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "ROC-AUC": np.nan,
        }
        try:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_eval)
                if y_score.shape[1] == 2:
                    metrics["ROC-AUC"] = roc_auc_score(y_true, y_score[:, 1])
                else:
                    metrics["ROC-AUC"] = roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted")
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_eval)
                metrics["ROC-AUC"] = roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted")
        except ValueError:
            pass
        return metrics, y_pred

    metrics = {
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
    return metrics, y_pred


def handle_missing_for_modeling(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    low_nan_row_threshold: float = 0.01,
):
    """
    Train-safe missing handling for modeling:
    1) If NaN rows are very low, drop those rows.
    2) Otherwise impute using train statistics (mean for numeric, mode for categorical).
    """
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()
    y_train_clean = y_train.copy()
    y_test_clean = y_test.copy()

    summary = {
        "dropped_train_rows": 0,
        "dropped_test_rows": 0,
        "imputed_numeric_cols": 0,
        "imputed_categorical_cols": 0,
        "dropped_all_nan_cols": [],
    }

    if X_train_clean.empty:
        return X_train_clean, X_test_clean, y_train_clean, y_test_clean, summary

    # Remove columns that are fully NaN in train split.
    all_nan_cols = X_train_clean.columns[X_train_clean.isna().all()].tolist()
    if all_nan_cols:
        X_train_clean = X_train_clean.drop(columns=all_nan_cols)
        X_test_clean = X_test_clean.drop(columns=all_nan_cols, errors="ignore")
        summary["dropped_all_nan_cols"] = all_nan_cols

    if X_train_clean.empty:
        return X_train_clean, X_test_clean, y_train_clean, y_test_clean, summary

    train_nan_rows_mask = X_train_clean.isna().any(axis=1)
    train_nan_ratio = train_nan_rows_mask.mean() if len(train_nan_rows_mask) else 0.0

    if train_nan_ratio > 0 and train_nan_ratio <= low_nan_row_threshold:
        keep_mask = ~train_nan_rows_mask
        summary["dropped_train_rows"] = int(train_nan_rows_mask.sum())
        X_train_clean = X_train_clean.loc[keep_mask].copy()
        y_train_clean = y_train_clean.loc[keep_mask].copy()
    elif train_nan_ratio > 0:
        num_cols = X_train_clean.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in X_train_clean.columns if c not in num_cols]

        for col in num_cols:
            if X_train_clean[col].isna().any() or X_test_clean[col].isna().any():
                fill_value = X_train_clean[col].mean()
                if pd.isna(fill_value):
                    fill_value = 0.0
                X_train_clean[col] = X_train_clean[col].fillna(fill_value)
                X_test_clean[col] = X_test_clean[col].fillna(fill_value)
                summary["imputed_numeric_cols"] += 1

        for col in cat_cols:
            if X_train_clean[col].isna().any() or X_test_clean[col].isna().any():
                mode_series = X_train_clean[col].mode(dropna=True)
                fill_value = mode_series.iloc[0] if not mode_series.empty else "missing"
                X_train_clean[col] = X_train_clean[col].fillna(fill_value)
                X_test_clean[col] = X_test_clean[col].fillna(fill_value)
                summary["imputed_categorical_cols"] += 1

    # Test split: if very low NaN rows remain, drop those rows + matching y_test.
    test_nan_rows_mask = X_test_clean.isna().any(axis=1)
    test_nan_ratio = test_nan_rows_mask.mean() if len(test_nan_rows_mask) else 0.0
    if test_nan_ratio > 0 and test_nan_ratio <= low_nan_row_threshold:
        keep_test_mask = ~test_nan_rows_mask
        summary["dropped_test_rows"] = int(test_nan_rows_mask.sum())
        X_test_clean = X_test_clean.loc[keep_test_mask].copy()
        y_test_clean = y_test_clean.loc[keep_test_mask].copy()

    # Final guard: fill any residual NaN by train stats.
    for col in X_train_clean.columns:
        if X_train_clean[col].isna().any() or (col in X_test_clean.columns and X_test_clean[col].isna().any()):
            if pd.api.types.is_numeric_dtype(X_train_clean[col]):
                fill_value = X_train_clean[col].mean()
                if pd.isna(fill_value):
                    fill_value = 0.0
            else:
                mode_series = X_train_clean[col].mode(dropna=True)
                fill_value = mode_series.iloc[0] if not mode_series.empty else "missing"
            X_train_clean[col] = X_train_clean[col].fillna(fill_value)
            if col in X_test_clean.columns:
                X_test_clean[col] = X_test_clean[col].fillna(fill_value)

    return X_train_clean, X_test_clean, y_train_clean, y_test_clean, summary


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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg-main: #f5f8ff;
    --bg-card: #ffffff;
    --line: #dbe4ff;
    --text: #16213e;
    --muted: #4f5f85;
    --brand: #5b7cfa;
    --brand-2: #10b981;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text);
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 10% 0%, #dbe7ff 0%, transparent 40%),
        radial-gradient(circle at 100% 100%, #d8fff1 0%, transparent 45%),
        var(--bg-main);
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: #11162a;
    border-right: 1px solid #2a345a;
}
[data-testid="stSidebar"] * { color: #e8edff !important; }
[data-testid="stSidebar"] code {
    color: #c6d2ff !important;
    background: #1a2342 !important;
}

.block-container {
    max-width: 1420px;
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4, p, li, span, div { color: var(--text); }
small { color: var(--muted); }

[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.75rem 0.9rem;
    box-shadow: 0 6px 20px rgba(60, 80, 160, 0.08);
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(35, 56, 128, 0.08);
}

.stTabs [role="tablist"] button {
    border-radius: 999px !important;
    border: 1px solid var(--line) !important;
    margin-right: 0.35rem;
    background: #fff !important;
    color: var(--muted) !important;
}
.stTabs [aria-selected="true"] {
    color: #fff !important;
    border-color: #4f6ff0 !important;
    background: linear-gradient(135deg, #5b7cfa 0%, #7a5af8 100%) !important;
}

.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #131b34 0%, #26335f 58%, #374f9f 100%);
    border-radius: 22px;
    padding: 1.7rem 1.9rem;
    border: 1px solid rgba(181, 199, 255, 0.25);
    box-shadow: 0 18px 40px rgba(24, 36, 74, 0.35);
    margin-bottom: 1rem;
}
.hero::after {
    content: "";
    position: absolute;
    right: -80px;
    top: -90px;
    width: 260px;
    height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.25) 0%, rgba(255,255,255,0) 72%);
}
.hero-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #d6e2ff;
    border: 1px solid rgba(214, 226, 255, 0.4);
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    padding: 0.28rem 0.62rem;
}
.hero-title {
    margin: 0.7rem 0 0.45rem 0;
    font-size: 2.05rem;
    line-height: 1.15;
    color: #ffffff;
    font-weight: 700;
}
.hero-sub {
    margin: 0;
    max-width: 76ch;
    color: #dae4ff;
    line-height: 1.65;
    font-size: 0.96rem;
}
.hero-chip {
    display: inline-block;
    margin-top: 0.9rem;
    border-radius: 10px;
    background: rgba(16,185,129,0.16);
    border: 1px solid rgba(16,185,129,0.45);
    color: #d6ffe9;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    padding: 0.24rem 0.58rem;
}

.section-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--bg-card);
    border: 1px solid var(--line);
    border-left: 6px solid var(--brand);
    border-radius: 14px;
    padding: 0.66rem 0.86rem;
    margin: 1.05rem 0 0.62rem;
    font-size: 1.02rem;
    font-weight: 700;
    box-shadow: 0 8px 24px rgba(43, 66, 150, 0.08);
}
.section-icon {
    width: 1.45rem;
    height: 1.45rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    background: rgba(91, 124, 250, 0.12);
}

.sidebar-panel {
    border-radius: 12px;
    border: 1px solid #2f3f73;
    background: linear-gradient(180deg, #181f3d 0%, #141a32 100%);
    padding: 0.78rem 0.82rem;
    margin: 0.65rem 0 0.9rem 0;
    font-size: 0.83rem;
    line-height: 1.55;
}
.control-card {
    border-radius: 12px;
    border: 1px solid #2f426e;
    background: #1a2243;
    padding: 0.62rem 0.72rem;
    margin: 0.35rem 0 0.55rem 0;
}
.control-card strong { color: #e6edff; }
.control-card small { color: #b9c6f1; }

.train-box {
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.88rem 1rem;
    margin: 0.72rem 0 0.9rem 0;
    box-shadow: 0 10px 24px rgba(71, 96, 184, 0.08);
}
.train-running { border-left: 4px solid var(--brand); }
.train-complete { border-left: 4px solid var(--brand-2); }
.train-box-title {
    color: #2a3f74;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.79rem;
}
.train-box-log {
    color: #3e4f79;
    font-size: 0.84rem;
    line-height: 1.55;
}

.step-card {
    background: var(--bg-card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.62rem;
    box-shadow: 0 8px 26px rgba(56, 75, 146, 0.08);
}
.step-card h4 { margin: 0 0 0.2rem 0; color: #1e2a53; }
.step-card p { margin: 0; color: #5f7098; font-size: 0.86rem; }
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.6rem;
    height: 1.6rem;
    border-radius: 999px;
    background: linear-gradient(135deg, #5b7cfa 0%, #7a5af8 100%);
    color: #fff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    margin-right: 0.48rem;
}

.ai-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #4059ad;
}
.ai-insight {
    border-radius: 14px;
    border: 1px solid #cfdbff;
    background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
    color: #2e3f6e;
    padding: 1rem 1.15rem;
    margin-top: 0.62rem;
    box-shadow: 0 8px 22px rgba(73, 100, 188, 0.08);
}

.badge-green {
    background: rgba(16,185,129,0.14);
    color: #0f8b66;
    border: 1px solid rgba(16,185,129,0.36);
    border-radius: 8px;
    padding: 0.14rem 0.52rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
}
.badge-orange {
    background: rgba(251,146,60,0.14);
    color: #b45309;
    border: 1px solid rgba(251,146,60,0.38);
    border-radius: 8px;
    padding: 0.14rem 0.52rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
}
.code-tag {
    background: #eef3ff;
    color: #3b4f95;
    border: 1px solid #d4defc;
    border-radius: 7px;
    padding: 0.1rem 0.42rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
}

.soft-divider {
    border: none;
    border-top: 1px dashed #c8d5ff;
    margin: 1.2rem 0;
}

[data-testid="stFileUploader"] section {
    background: #fff;
    border: 2px dashed #c6d6ff;
    border-radius: 14px;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #7f9aff;
    background: #f8faff;
}
[data-testid="stButton"] > button {
    border-radius: 11px;
    border: 1px solid #5f7ffb;
    background: linear-gradient(135deg, #5b7cfa 0%, #7a5af8 100%);
    color: #fff;
    font-weight: 600;
}
[data-testid="stButton"] > button:hover {
    border-color: #3e63f7;
    box-shadow: 0 0 0 3px rgba(91,124,250,0.2);
}
</style>
""", unsafe_allow_html=True)

# Premium visual override
st.markdown("""
<style>
:root {
    --lux-bg-0: #070914;
    --lux-bg-1: #0d1226;
    --lux-bg-2: #161a34;
    --lux-glass: rgba(255, 255, 255, 0.08);
    --lux-line: rgba(176, 196, 255, 0.25);
    --lux-text: #f1f5ff;
    --lux-muted: #b6c2e8;
    --lux-brand: #7ea2ff;
    --lux-accent: #9b8cff;
}

[data-testid="stAppViewContainer"] {
    background:
      radial-gradient(circle at 8% 12%, rgba(126,162,255,0.28) 0%, transparent 34%),
      radial-gradient(circle at 92% 8%, rgba(155,140,255,0.24) 0%, transparent 38%),
      radial-gradient(circle at 50% 100%, rgba(30,214,160,0.14) 0%, transparent 30%),
      linear-gradient(145deg, var(--lux-bg-0) 0%, var(--lux-bg-1) 45%, var(--lux-bg-2) 100%) !important;
}

h1, h2, h3, h4, p, li, span, div, label {
    color: var(--lux-text) !important;
}
small, [data-testid="stCaptionContainer"] {
    color: var(--lux-muted) !important;
}

.hero {
    background:
      linear-gradient(130deg, rgba(17,24,48,0.86) 0%, rgba(32,35,71,0.84) 52%, rgba(47,36,88,0.82) 100%) !important;
    border: 1px solid var(--lux-line) !important;
    border-radius: 26px !important;
    box-shadow:
      0 24px 50px rgba(0,0,0,0.5),
      inset 0 1px 0 rgba(255,255,255,0.18) !important;
    backdrop-filter: blur(8px);
}
.hero-badge {
    color: #e9efff !important;
    background: rgba(126,162,255,0.2) !important;
    border: 1px solid rgba(154,180,255,0.55) !important;
}
.hero-title {
    font-size: 2.35rem !important;
    letter-spacing: -0.6px !important;
    color: #ffffff !important;
    text-shadow: 0 2px 14px rgba(126,162,255,0.28);
}
.hero-sub {
    color: #dbe4ff !important;
    font-size: 1rem !important;
}
.hero-chip {
    background: rgba(30,214,160,0.18) !important;
    border: 1px solid rgba(30,214,160,0.5) !important;
    color: #d8fff1 !important;
}

.section-title,
[data-testid="stMetric"],
[data-testid="stDataFrame"],
.step-card,
.ai-insight,
.train-box,
[data-testid="stExpander"] {
    background: var(--lux-glass) !important;
    border: 1px solid var(--lux-line) !important;
    backdrop-filter: blur(8px);
    box-shadow: 0 14px 34px rgba(0,0,0,0.28) !important;
}

.section-title {
    border-left: 6px solid var(--lux-brand) !important;
}
.section-icon {
    background: rgba(126,162,255,0.18) !important;
}

.step-card p,
.train-box-log,
.ai-insight {
    color: #d3defd !important;
}
.ai-label {
    color: #9cb8ff !important;
}

.stTabs [role="tablist"] button {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid var(--lux-line) !important;
    color: #d9e3ff !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7ea2ff 0%, #9b8cff 100%) !important;
    color: #fff !important;
}

[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.06) !important;
    border: 2px dashed rgba(164,184,255,0.55) !important;
}
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7ea2ff 0%, #9b8cff 100%) !important;
    border: 1px solid rgba(201,214,255,0.65) !important;
    box-shadow: 0 8px 24px rgba(88,112,220,0.45);
}
[data-testid="stButton"] > button:hover {
    box-shadow: 0 0 0 3px rgba(147,171,255,0.28), 0 12px 28px rgba(88,112,220,0.55) !important;
}

.badge-green {
    background: rgba(30,214,160,0.2) !important;
    border: 1px solid rgba(30,214,160,0.55) !important;
    color: #d8fff1 !important;
}
.badge-orange {
    background: rgba(255,171,74,0.2) !important;
    border: 1px solid rgba(255,171,74,0.58) !important;
    color: #ffe5bf !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────
ai_mode_badge = "AI-Assisted Mode" if AI_ENABLED else "Local-Only Mode"
st.markdown(f"""
<div class="hero">
  <div class="hero-badge">AURORA ANALYTICS SUITE · OPENROUTER NATIVE</div>
  <h1 class="hero-title">🧬 Preprocessing Atelier</h1>
  <p class="hero-sub">
    A richer, cinematic workspace for end-to-end tabular intelligence.
    Upload your dataset, shape a refined preprocessing narrative, and export production-grade outputs with AI context.
  </p>
  <div class="hero-chip">{ai_mode_badge}</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# SIDEBAR — settings
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Controls")
    st.markdown(
        f"""
        <div class="sidebar-panel">
            <strong>Execution Profile</strong><br>
            Model: <code>z-ai/glm-4.5-air:free</code><br>
            AI: <strong>{"Enabled" if AI_ENABLED else "Disabled"}</strong><br>
            Engine: <strong>pandas + scikit-learn</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
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
    encoding_method = st.selectbox("Categorical encoding", ["One-Hot Encoding", "Label Encoding", "Both"], index=0, key="encoding_method")
    test_size = st.slider("Test split size", 0.10, 0.40, 0.20, 0.05, key="test_size")
    max_onehot_unique = st.slider("Max unique values for one-hot", 2, 30, 10, 1, key="max_onehot_unique")
    st.markdown(
        """
        <div class="control-card">
            <strong>🧪 Model Training Controls</strong>
            <small>Controls for cross-validation and random-search tuning intensity.</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    model_cv_folds_cfg = st.slider("CV folds", 2, 10, 5, 1, key="sidebar_model_cv_folds")
    model_tuning_iterations_cfg = st.slider(
        "Tuning iterations (RandomizedSearchCV)",
        1,
        50,
        12,
        1,
        key="sidebar_model_tuning_iterations",
    )

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
# PRE-UPLOAD CONTEXT (required before upload & processing)
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Pre-Upload Context", "📝")
st.markdown("Fill these first, then upload dataset. OpenRouter uses this context to decide supervised vs unsupervised.")

ctx_col1, ctx_col2 = st.columns(2)
dataset_description_text = ctx_col1.text_area(
    "🗂️ Dataset Description",
    value=st.session_state.get("dataset_description_text", ""),
    placeholder="Describe dataset source, entities, and important columns.",
    key="dataset_description_text",
    height=110,
)
problem_statement_text = ctx_col2.text_area(
    "🎯 Problem Statement",
    value=st.session_state.get("problem_statement_text", ""),
    placeholder="Example: Predict churn / Segment customers / Detect anomalies.",
    key="problem_statement_text",
    height=110,
)
st.markdown("#### ✅ Current Context")
ctx_show_1, ctx_show_2 = st.columns(2)
ctx_show_1.markdown(f"**Dataset Description:**\n\n{dataset_description_text.strip() or '_Not provided yet_'}")
ctx_show_2.markdown(f"**Problem Statement:**\n\n{problem_statement_text.strip() or '_Not provided yet_'}")

# ─────────────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Upload Dataset (CSV)",
    type=["csv"],
    help="Upload a tabular CSV dataset to initialize the full preprocessing pipeline.",
)

if uploaded_file is None:
    st.markdown("""
    <div class="step-card">
      <h4>📋 End-to-End EDA + Feature Engineering Pipeline</h4>
      <p>
        Upload CSV → guided preprocessing with <strong>inline visual diagnostics in every stage</strong> →
        export cleaned data and reproducible pipeline code.
      </p>
      <p><strong>Includes outlier diagnostics, encoding strategy, and train/test readiness checks.</strong></p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    steps_left = [
        ("1", "Problem Type Detection", "supervised vs unsupervised inference"),
        ("2", "Data Overview", "shape, schema, head, descriptive stats"),
        ("3", "Data Cleaning", "missing values + duplicate handling"),
        ("4", "Column Categorization", "continuous, discrete, categorical, datetime"),
        ("5", "Univariate Analysis", "distributions, skewness, category frequencies"),
        ("6", "Conditional Analysis", "target-aware analysis if supervised"),
        ("7", "Multivariate Analysis", "correlation, scatter, pair relationships"),
        ("8", "Outlier Handling", "IQR/Z-score with cap/remove strategy"),
    ]
    steps_right = [
        ("9", "Feature Engineering", "transformations and derived features"),
        ("10", "Encoding", "label/one-hot/frequency decisions"),
        ("11", "Feature Selection", "variance + high-correlation filters"),
        ("12", "Final Preparation", "train/test split and export"),
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
# DROP ONLY PRIMARY ID COLUMNS (not all high-unique columns)
# ─────────────────────────────────────────────────────────
id_columns = detect_primary_id_columns(df)
if id_columns:
    df.drop(columns=id_columns, inplace=True)
    st.info(f"🗑️ Dropped primary ID column(s): {', '.join(id_columns)}")

# ─────────────────────────────────────────────────────────
# WORKFLOW OVERVIEW
# ─────────────────────────────────────────────────────────
st.markdown("## 🔄 Complete EDA & Feature Engineering Workflow")
st.markdown("""
**⚠️ Important: All visualizations are generated INLINE within each step!**

**Prompt-Aligned Pipeline (Auto-skip when not relevant):**
1. Problem Type Detection
2. Data Overview
3. Data Cleaning
4. Column Categorization
5. Univariate Analysis
6. Conditional Analysis (Supervised/Unsupervised)
7. Multivariate Analysis
8. Outlier Detection & Handling
9. Feature Engineering
10. Encoding
11. Feature Selection
12. Final Dataset Preparation

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
3. FEATURE ENGINEERING: Suggest encoding methods
4. FEATURE SELECTION: Recommend variance threshold, correlation handling
5. TRAIN/TEST SPLIT: Suggest optimal split ratio

Return JSON with these exact keys:
- missing_strategy: "Auto (smart fill)" | "Fill — mean" | "Fill — median" | "Fill — mode" | "Drop rows"
- outlier_method: "IQR" | "Z-score"
- outlier_action: "Cap (clip)" | "Remove rows"
- z_threshold: 2.0-4.0
- iqr_factor: 1.0-3.0
- encoding_method: "One-Hot Encoding" | "Label Encoding" | "Both"
- test_size: 0.1-0.4
- max_onehot_unique: 2-30
- reasoning: brief explanation of workflow decisions

Decision Rules:
- Use mean for normally distributed data, median for skewed
- IQR for small datasets (<1000), Z-score for large (>1000)
- Cap outliers if <5% of data, remove if >5%
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
# STEP 1 — Problem Type Detection
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 1 — Problem Type Detection", "🎯")

st.markdown("""
Identify the **Target Variable** to determine if this is a Binary Classification 
or Multi-Class (3-4+ types) problem. This helps in handling imbalanced data later.
""")

# Basic EDA snapshot before OpenRouter decision
basic_eda_summary = {
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "numeric_columns": int(len(df.select_dtypes(include="number").columns)),
    "categorical_columns": int(len(df.select_dtypes(include=["object", "category", "bool"]).columns)),
    "missing_cells": int(df.isnull().sum().sum()),
    "missing_columns": [c for c in df.columns if df[c].isnull().any()][:20],
    "top_unique_columns": {c: int(df[c].nunique(dropna=True)) for c in df.columns[:15]},
}
st.markdown("#### Basic EDA Snapshot (used for AI decision)")
eda_c1, eda_c2, eda_c3, eda_c4 = st.columns(4)
eda_c1.metric("Rows", f"{basic_eda_summary['rows']:,}")
eda_c2.metric("Columns", basic_eda_summary["columns"])
eda_c3.metric("Numeric", basic_eda_summary["numeric_columns"])
eda_c4.metric("Categorical", basic_eda_summary["categorical_columns"])




with st.expander("View EDA stats sent to OpenRouter"):
    st.code(pretty_json(basic_eda_summary), language="json")

# OpenRouter AI dataset analysis for supervised/unsupervised + target recommendation
dataset_signature = (
    f"{uploaded_file.name}|{df.shape[0]}|{df.shape[1]}|"
    f"{','.join(df.columns.tolist())}|{int(df.isnull().sum().sum())}|"
    f"{dataset_description_text.strip()}|{problem_statement_text.strip()}"
)
ai_target_key = "ai_target_analysis_result"
ai_target_sig_key = "ai_target_analysis_signature"
previous_ai_signature = st.session_state.get(ai_target_sig_key)

if AI_ENABLED and st.session_state.get(ai_target_sig_key) != dataset_signature:
    try:
        with st.spinner("🤖 OpenRouter is inferring supervised vs unsupervised and target column candidates..."):
            st.session_state[ai_target_key] = ai_target_detection(
                df,
                dataset_description=dataset_description_text,
                problem_statement=problem_statement_text,
                basic_eda_summary=basic_eda_summary,
            )
            st.session_state[ai_target_sig_key] = dataset_signature
    except Exception as exc:
        st.session_state[ai_target_key] = None
        st.session_state[ai_target_sig_key] = dataset_signature
        st.warning(f"⚠️ OpenRouter target analysis failed: {exc}. Falling back to heuristics with your context.")

ai_target_analysis = st.session_state.get(ai_target_key) if AI_ENABLED else None
if st.session_state.get("selected_target") not in df.columns:
    st.session_state["selected_target"] = None
if AI_ENABLED and ai_target_analysis and previous_ai_signature != dataset_signature:
    if ai_target_analysis["learning_type"] == "unsupervised":
        st.session_state["selected_target"] = None
    elif ai_target_analysis["recommended_target"] in df.columns:
        st.session_state["selected_target"] = ai_target_analysis["recommended_target"]

if AI_ENABLED and ai_target_analysis:
    ai_lines = [
        f"**Learning type:** `{ai_target_analysis['learning_type'].title()}`",
        f"**Task hint:** `{ai_target_analysis['task_type'].title()}`",
        f"**Recommended target:** `{ai_target_analysis['recommended_target'] or 'None (Unsupervised)'}`",
        f"**Confidence:** `{ai_target_analysis['confidence']:.2f}`",
    ]
    if ai_target_analysis["alternative_targets"]:
        ai_lines.append(
            f"**Alternative target candidates:** `{', '.join(ai_target_analysis['alternative_targets'])}`"
        )
    if ai_target_analysis["reasoning"]:
        ai_lines.append(f"**Reasoning:** {ai_target_analysis['reasoning']}")
    if ai_target_analysis["cautions"]:
        ai_lines.append(f"**Cautions:** {ai_target_analysis['cautions']}")
    st.markdown(
        f'<div class="ai-label">🤖 OPENROUTER DATASET ANALYSIS</div>'
        f'<div class="ai-insight">{"<br><br>".join(ai_lines)}</div>',
        unsafe_allow_html=True,
    )
elif not AI_ENABLED:
    st.info("ℹ️ OpenRouter AI is disabled. Enable API key to auto-detect supervised/unsupervised and target candidates.")

# Heuristic fallback for candidate targets
potential_targets = df.select_dtypes(include=['object']).columns.tolist()
low_card_numeric = [c for c in df.select_dtypes(include='number').columns if df[c].nunique() <= 15]
potential_targets = list(set(potential_targets + low_card_numeric))

default_selection = df.columns[-1]
if AI_ENABLED and ai_target_analysis:
    if ai_target_analysis["learning_type"] == "unsupervised":
        default_selection = None
    elif ai_target_analysis["recommended_target"] in df.columns:
        default_selection = ai_target_analysis["recommended_target"]
elif default_selection not in potential_targets and len(potential_targets) > 0:
    default_selection = potential_targets[0]

# Selectbox for Target Selection (with unsupervised option)
target_options = ["None (Unsupervised)"] + df.columns.tolist()
default_target = st.session_state.get("selected_target", default_selection)
if default_target is None:
    target_index = 0
else:
    target_index = target_options.index(default_target) if default_target in target_options else 1

target_col = st.selectbox(
    "👉 Select your **Target Variable** (choose 'None' for unsupervised):",
    options=target_options,
    index=target_index,
    key="target_column_selector"
)

if target_col == "None (Unsupervised)":
    problem_type = "Unsupervised (Clustering / Pattern Discovery)"
    st.session_state['selected_target'] = None
    st.session_state['problem_type'] = problem_type
    st.markdown(f'<span class="badge-green">{problem_type}</span>', unsafe_allow_html=True)
    st.info("No target selected. Pipeline will continue in unsupervised-friendly mode where applicable.")

    ai_step_1 = ask_deepseek(
        "You are a data scientist. Explain what unsupervised analysis means for preprocessing and EDA. "
        "Give concise, practical guidance in 4-6 bullets.",
        f"Dataset has no selected target. Shape: {df.shape}. Numeric columns: {len(df.select_dtypes(include='number').columns)}"
    )
    st.markdown(f'<div class="ai-label">🤖 AI ANALYSIS</div><div class="ai-insight">{ai_step_1}</div>', unsafe_allow_html=True)
else:
    unique_classes = df[target_col].nunique(dropna=True)
    class_counts = df[target_col].value_counts(dropna=False)
    
    # Determine Problem Type & Badge Color
    if pd.api.types.is_numeric_dtype(df[target_col]) and unique_classes > 15:
        problem_type = "Regression"
        badge_color = "badge-orange"
    elif unique_classes == 2:
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
    st.markdown("#### 📊 Target Distribution")
    fig, ax = plt.subplots(figsize=(max(8, min(unique_classes, 20)), 5))
    
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

    # Store target in session state for later steps (like splitting)
    st.session_state['selected_target'] = target_col
    st.session_state['problem_type'] = problem_type

    # AI Insight
    ai_step_1 = ask_deepseek(
        "You are a data scientist. Analyze the class distribution of the target variable. "
        "1) Is the dataset balanced or imbalanced? "
        "2) If imbalanced, suggest 2 techniques to handle it (e.g., SMOTE, Class weights). "
        "3) Confirm whether this is classification or regression.",
        f"Target: {target_col}, Problem type: {problem_type}, Unique values: {unique_classes}, Counts: {class_counts.head(20).to_dict()}"
    )
    st.markdown(f'<div class="ai-label">🤖 AI ANALYSIS</div><div class="ai-insight">{ai_step_1}</div>', unsafe_allow_html=True)

is_unsupervised_mode = st.session_state.get("selected_target") is None
if is_unsupervised_mode:
    st.info("🧭 Mode selected: **Unsupervised workflow** (pattern discovery, PCA, clustering).")
else:
    st.info("🧭 Mode selected: **Supervised workflow** (target-driven EDA, feature engineering, model training).")

# ─────────────────────────────────────────────────────────
# STEP 2 — Data Overview
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 2 — Data Overview", "📊")

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
    describe_df = df.copy()
    obj_like_cols = describe_df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in obj_like_cols:
        describe_df[col] = describe_df[col].map(lambda v: str(v) if pd.notna(v) else np.nan)

    try:
        describe_out = describe_df.describe(include="all").T
        st.dataframe(describe_out, use_container_width=True)
    except (IndexError, TypeError, ValueError) as exc:
        st.warning(f"⚠️ Full describe failed for mixed object columns: {exc}")
        numeric_only = describe_df.select_dtypes(include=["number"])
        if numeric_only.shape[1] > 0:
            st.markdown("**Numeric summary (safe fallback):**")
            st.dataframe(numeric_only.describe().T, use_container_width=True)
        else:
            st.info("ℹ️ No numeric columns available for fallback describe.")

schema_summary = {
    "shape": list(df.shape),
    "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    "missing_count": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
    "unique_counts": {c: int(df[c].nunique()) for c in df.columns},
}
ai_step2 = ask_deepseek(
    "You are a senior ML engineer. Analyse this dataset summary and highlight "
    "the most important observations for preprocessing. Be concise (5-8 bullet points). "
    "Mention data types, missing values, cardinality issues, and potential preprocessing challenges.",
    f"Dataset schema:\n{pretty_json(schema_summary)}",
)
st.markdown(f'<div class="ai-label">🤖 AI ANALYSIS</div><div class="ai-insight">{ai_step2}</div>', unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────
# STEP 3 — Data Cleaning: Missing values
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 3 — Data Cleaning: Missing Values", "🩹")

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

    ai_step3_missing = ask_deepseek(
        "You are a senior ML engineer. Explain the missing-value handling decisions below "
        "in simple terms. Mention why median is preferred over mean for skewed data, "
        "mode for categoricals, and when to drop. Use 4–6 bullet points.",
        f"Actions taken:\n{pretty_json(fill_log)}\nDataset shape after: {df.shape}",
    )
    st.markdown(f'<div class="ai-label">🤖 AI EXPLANATION</div><div class="ai-insight">{ai_step3_missing}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# STEP 3 — Data Cleaning: Duplicates
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 3 — Data Cleaning: Duplicate Rows", "🔁")

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

ai_step3_dupes = ask_deepseek(
    "You are a senior ML engineer. Explain in 3 bullet points why removing duplicates "
    "is critical before training an ML model, and what risks arise if not done.",
    f"Duplicates removed: {n_dupes}. Remaining rows: {len(df)}",
)
st.markdown(f'<div class="ai-label">🤖 AI EXPLANATION</div><div class="ai-insight">{ai_step3_dupes}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# STEP 4 — Column Categorization
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 4 — Column Categorization (Auto-Detect)", "🧩")

# Separate Features
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()
datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

# Identify discrete vs continuous numeric features
discrete_cols = [col for col in num_cols if df[col].nunique() < 20]
continuous_cols = [col for col in num_cols if df[col].nunique() >= 20]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Continuous", len(continuous_cols))
c2.metric("Discrete", len(discrete_cols))
c3.metric("Categorical", len(cat_cols))
c4.metric("Datetime", len(datetime_cols))

st.caption("Columns are auto-categorized to drive downstream analysis and preprocessing decisions.")

# ─────────────────────────────────────────────────────────
# STEP 5 — Univariate Analysis
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 5 — Univariate Analysis", "📊")

st.markdown("#### Analyzing Individual Features")

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
        col_series = safe_numeric_series(df[col])
        
        # 1. Histogram & KDE
        if col_series.empty:
            ax.text(0.5, 0.5, "No valid numeric values", ha="center", va="center")
            ax.set_title(col, fontsize=10)
            ax.set_axis_off()
            continue
        ok_plot = plot_hist_kde_safe(ax, col_series, color="skyblue", alpha=0.6)
        if not ok_plot:
            ax.text(0.5, 0.5, "Histogram unavailable", ha="center", va="center")
            ax.set_axis_off()
            continue
        ax.set_title(f'{col}\nSkew: {col_series.skew():.2f}', fontsize=10)
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
        col_series = safe_numeric_series(df[col])
        if col_series.empty:
            axes2[idx].text(0.5, 0.5, "No valid numeric values", ha="center", va="center")
            axes2[idx].set_axis_off()
            continue
        sns.boxplot(x=col_series, ax=axes2[idx], color='lightcoral')
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
        col_counts = safe_value_counts(df[col], normalize=False)
        col_mode = safe_categorical_series(df[col]).mode(dropna=True)
        discrete_stats.append({
            "Feature": col,
            "Unique Values": df[col].nunique(),
            "Most Common": col_mode.iloc[0] if not col_mode.empty else "N/A",
            "Frequency": int(col_counts.iloc[0]) if len(col_counts) > 0 else 0
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
        counts = safe_value_counts(df[col], normalize=False)
        if counts.empty:
            ax.text(0.5, 0.5, "No valid values", ha="center", va="center")
            ax.set_axis_off()
            continue
        counts.plot(kind='bar', ax=ax, color='teal', alpha=0.7)
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
        col_series = safe_categorical_series(df[col])
        val_counts = safe_value_counts(col_series, normalize=True)
        rare_count = (val_counts < 0.05).sum() # Count categories with less than 5% frequency
        col_mode = col_series.mode(dropna=True)
        
        cat_stats.append({
            "Feature": col,
            "Unique Values": col_series.nunique(dropna=True),
            "Most Common": col_mode.iloc[0] if not col_mode.empty else "N/A",
            "Rare Categories (<5%)": rare_count,
            "Type": "High Cardinality" if col_series.nunique(dropna=True) > 10 else "Low Cardinality"
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
        col_series = safe_categorical_series(df[col])
        counts = safe_value_counts(col_series, normalize=False)
        if counts.empty:
            ax.text(0.5, 0.5, "No valid categorical values", ha="center", va="center")
            ax.set_axis_off()
            continue
        counts = counts.head(20)
        ax.bar(counts.index.astype(str), counts.values, color="teal", alpha=0.8)
        ax.set_title(f'Count Plot: {col}', fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
    
    for idx in range(len(cat_cols[:4]), len(axes)):
        fig.delaxes(axes[idx])
    plt.tight_layout()
    st.pyplot(fig)


# ─────────────────────────────────────────────────────────
# STEP 6 & 7 — Conditional + Multivariate Analysis
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 6 & 7 — Conditional + Multivariate Analysis", "🔗")

st.markdown("#### Analyzing Relationships Between Features")
if is_unsupervised_mode:
    st.caption("Unsupervised emphasis: feature interactions and latent structure discovery (no target dependence).")
else:
    st.caption("Supervised emphasis: target-aware relationship diagnostics for predictive modeling.")

# Recompute columns from current dataframe to keep Step 6–7 always up-to-date
step67_num_cols = df.select_dtypes(include="number").columns.tolist()
step67_cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# 🔢 Numerical vs Numerical
if len(step67_num_cols) > 1:
    st.markdown("### 🔢 Numerical vs Numerical")
    st.markdown("**Analysis:** Relationship between continuous variables using 5 complementary plots.")

    corr_matrix = df[step67_num_cols].corr()

    # 1) Correlation Heatmap
    st.markdown("**1) Correlation Heatmap**")
    fig, ax = plt.subplots(figsize=(12, 10))
    annot_flag = len(step67_num_cols) <= 12
    sns.heatmap(
        corr_matrix,
        annot=annot_flag,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)

    # 2) Correlation Strength Bar Plot
    st.markdown("**2) Correlation Strength Bar Plot (avg |correlation| per feature)**")
    corr_abs = corr_matrix.abs().copy()
    np.fill_diagonal(corr_abs.values, np.nan)
    corr_strength = corr_abs.mean().sort_values(ascending=False)
    if corr_strength.dropna().shape[0] > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(x=corr_strength.index, y=corr_strength.values, ax=ax, color="#62b8ff")
        ax.set_title("Average Absolute Correlation by Numeric Feature")
        ax.set_xlabel("Numeric Features")
        ax.set_ylabel("Avg |r|")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    # 3) Histograms with KDE
    st.markdown("**3) Histograms + KDE (distribution view)**")
    hist_cols = step67_num_cols[: min(6, len(step67_num_cols))]
    n_cols = 3
    n_rows = (len(hist_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    for idx, col in enumerate(hist_cols):
        ax = axes[idx]
        col_series = safe_numeric_series(df[col])
        if col_series.empty:
            ax.text(0.5, 0.5, "No valid numeric values", ha="center", va="center")
            ax.set_title(f"Histogram: {col}")
            ax.set_axis_off()
            continue
        ok_plot = plot_hist_kde_safe(ax, col_series, color="mediumpurple")
        if not ok_plot:
            ax.text(0.5, 0.5, "Histogram unavailable", ha="center", va="center")
            ax.set_axis_off()
            continue
        ax.set_title(f"Histogram: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    for idx in range(len(hist_cols), len(axes)):
        fig.delaxes(axes[idx])
    plt.tight_layout()
    st.pyplot(fig)

    # 4) Top Correlated Pairs Scatter Plots
    st.markdown("**4) Top Correlated Pairs (scatter + trend line)**")
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], abs(corr_matrix.iloc[i, j])))
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = corr_pairs[: min(4, len(corr_pairs))]

    if top_pairs:
        n_cols = 2
        n_rows = (len(top_pairs) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        for idx, (col1, col2, corr_val) in enumerate(top_pairs):
            sns.scatterplot(data=df, x=col1, y=col2, ax=axes[idx], alpha=0.6, color="purple")
            sns.regplot(data=df, x=col1, y=col2, ax=axes[idx], scatter=False, color="orange", truncate=True)
            axes[idx].set_title(f"{col1} vs {col2} (|r|={corr_val:.2f})")

        for idx in range(len(top_pairs), len(axes)):
            fig.delaxes(axes[idx])
        plt.tight_layout()
        st.pyplot(fig)

    # 5) Pairplot (sampled if large)
    if 2 <= len(step67_num_cols) <= 6:
        st.markdown("**5) Pairplot (sampled, full pairwise relationship view)**")
        plot_df = df[step67_num_cols]
        if len(plot_df) > 1000:
            plot_df = plot_df.sample(1000, random_state=42)

        fig = sns.pairplot(plot_df, corner=True, plot_kws={"alpha": 0.6, "s": 25, "edgecolor": "k"})
        fig.fig.suptitle("Pairplot of Numeric Features", y=1.02)
        st.pyplot(fig)

# 🔢 Numerical vs Categorical
if len(step67_cat_cols) > 0 and len(step67_num_cols) > 0:
    st.markdown("### 🔢 Numerical vs Categorical")
    st.markdown("**Analysis:** Category-wise numeric behavior using bar + distribution plots.")

    # 1) Multi-pair mean bar charts
    candidate_cats = [c for c in step67_cat_cols if df[c].nunique() > 1][:2]
    candidate_nums = step67_num_cols[:2]
    pair_candidates = [(c, n) for c in candidate_cats for n in candidate_nums]

    if pair_candidates:
        st.markdown("**1) Mean Value by Category (Bar Charts)**")
        n_cols = 2
        n_rows = (len(pair_candidates) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        for idx, (cat_col, num_col) in enumerate(pair_candidates):
            ax = axes[idx]
            top_cats = df[cat_col].value_counts().head(12).index
            temp = df[df[cat_col].isin(top_cats)]
            bar_data = temp.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
            sns.barplot(x=bar_data.index, y=bar_data.values, ax=ax, palette="viridis")
            ax.set_title(f"Avg {num_col} by {cat_col}")
            ax.set_xlabel(cat_col)
            ax.set_ylabel(f"Mean {num_col}")
            ax.tick_params(axis="x", rotation=45)

        for idx in range(len(pair_candidates), len(axes)):
            fig.delaxes(axes[idx])
        plt.tight_layout()
        st.pyplot(fig)

    # 2) Box + Violin for primary pair
    target_cat = step67_cat_cols[0]
    target_num = step67_num_cols[0]
    st.markdown(f"**2) Box Plot + Violin Plot ({target_num} by {target_cat})**")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    top_cats = df[target_cat].value_counts().head(12).index
    plot_df = df[df[target_cat].isin(top_cats)].copy()

    sns.boxplot(x=plot_df[target_cat], y=plot_df[target_num], ax=axes[0], palette="Set2")
    axes[0].set_title(f"Box Plot: {target_num} by {target_cat}")
    axes[0].tick_params(axis="x", rotation=45)

    sns.violinplot(x=plot_df[target_cat], y=plot_df[target_num], ax=axes[1], palette="muted", inner="quartile")
    axes[1].set_title(f"Violin Plot: {target_num} by {target_cat}")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

# 🔤 Categorical vs Categorical
if len(step67_cat_cols) > 1:
    st.markdown("### 🔤 Categorical vs Categorical")
    st.markdown("**Analysis:** Relationship patterns across categorical columns with bar and heatmap.")

    cat_candidates = [c for c in step67_cat_cols if 1 < df[c].nunique() <= 12][:4]
    cat_pairs = []
    for i in range(len(cat_candidates)):
        for j in range(i + 1, len(cat_candidates)):
            cat_pairs.append((cat_candidates[i], cat_candidates[j]))
    cat_pairs = cat_pairs[:3]

    if cat_pairs:
        st.markdown("**1) Normalized Stacked Bar Charts (category proportions)**")
        fig, axes = plt.subplots(1, len(cat_pairs), figsize=(7 * len(cat_pairs), 6))
        if len(cat_pairs) == 1:
            axes = [axes]

        for idx, (cat1, cat2) in enumerate(cat_pairs):
            crosstab_norm = pd.crosstab(df[cat1], df[cat2], normalize="index")
            crosstab_norm.plot(kind="bar", stacked=True, ax=axes[idx], colormap="Paired", legend=False)
            axes[idx].set_title(f"{cat1} vs {cat2}")
            axes[idx].set_xlabel(cat1)
            axes[idx].set_ylabel("Proportion")
            axes[idx].tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        cat1, cat2 = cat_pairs[0]
        st.markdown(f"**2) Crosstab Heatmap ({cat1} vs {cat2})**")
        crosstab = pd.crosstab(df[cat1], df[cat2])
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(crosstab, annot=True, fmt="d", cmap="Blues", ax=ax, linewidths=0.5)
        ax.set_title(f"Crosstab Heatmap: {cat1} vs {cat2}")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

# ⚠️ Step 8 — Outlier Handling
st.markdown("### Step 8 — ⚠️ Outlier Detection & Handling")
st.markdown("**Method:** IQR (Interquartile Range) | **Decide:** Remove, Cap, Keep")

# Using Box Plots to identify outliers (Already shown in Step 3, but reiterating context here)
num_cols_now = df.select_dtypes(include="number").columns.tolist()
outlier_log = {}
rows_before = len(df)

# Apply Outlier Handling based on selected method/action

st.markdown("**Outlier Detection & Treatment Log:**")

for col in num_cols_now:
    col_data = df[col].dropna()
    if len(col_data) == 0: continue
    
    if outlier_method == "Z-score":
        col_std = col_data.std()
        if col_std == 0 or np.isnan(col_std):
            continue
        z_scores = np.abs((df[col] - col_data.mean()) / col_std)
        mask = z_scores > z_threshold
        lo, hi = col_data.mean() - z_threshold * col_std, col_data.mean() + z_threshold * col_std
    else:
        Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - iqr_factor * IQR, Q3 + iqr_factor * IQR
        mask = (df[col] < lo) | (df[col] > hi)

    n_out = int(mask.sum())
    
    if n_out > 0:
        if outlier_action == "Cap (clip)":
            action = "Cap (clip)"
            df[col] = df[col].clip(lower=lo, upper=hi)
            outlier_log[col] = {"outliers": n_out, "action": action, "details": f"Clipped at [{lo:.2f}, {hi:.2f}]"}
        else:
            action = "Remove"
            df = df[~mask]
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
# STEP 9–10 — Feature Engineering & Encoding
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 9–10 — Feature Engineering & Encoding", "⚙️")

st.markdown("#### Preparing data for Machine Learning")
if is_unsupervised_mode:
    st.caption("Unsupervised preprocessing emphasis: transformation quality and distance-friendly feature space for clustering.")
else:
    st.caption("Supervised preprocessing emphasis: predictive signal preservation with leakage-safe transformations.")

# 🔢 Numerical Features: Skewness & Scaling
st.markdown("### 🔢 Numerical Features Transformation")
st.markdown("**Goal:** Handle skewness while deferring scaling to model training")

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
        col_series = safe_numeric_series(df[col])
        if col_series.empty:
            ax.text(0.5, 0.5, "No valid numeric values", ha="center", va="center")
            ax.set_title(col)
            ax.set_axis_off()
            continue
        
        # Plot original
        plot_hist_kde_safe(ax, col_series, color="skyblue", label="Original", density=True, alpha=0.5)
        
        # Signed log transform is robust for both positive and negative values.
        transformed_data = np.sign(col_series) * np.log1p(np.abs(col_series))
        transformed_cols[col] = transformed_data # Store for potential later use
        
        plot_hist_kde_safe(
            ax,
            pd.Series(transformed_data),
            color="purple",
            label="Log Transformed",
            density=True,
            alpha=0.5,
        )
        
        ax.set_title(f'{col} (Skew: {col_series.skew():.2f} -> {transformed_data.skew():.2f})')
        ax.legend()
        
    for idx in range(len(high_skew[:3]), len(axes)):
        fig.delaxes(axes[idx])
        
    plt.tight_layout()
    st.pyplot(fig)
    
    # Apply transformations to dataframe for the next steps
    for col in high_skew:
        base_col = pd.to_numeric(df[col], errors="coerce")
        df[f'{col}_log'] = np.sign(base_col) * np.log1p(np.abs(base_col))

# 🔤 Categorical Features (Encoding)
st.markdown("### Step 10 — 🔤 Categorical Features (Encoding)")
st.markdown("**Strategy:** Label Encoding (Ordinal) vs One-Hot (Nominal)")

cat_cols_now = df.select_dtypes(include="object").columns.tolist()
encoding_map = {}

for col in cat_cols_now:
    unique_count = df[col].nunique()
    
    if encoding_method == "Label Encoding":
        encoding_map[col] = "Label Encoding"
        mapping = {k: i for i, k in enumerate(df[col].astype(str).unique())}
        df[col] = df[col].astype(str).map(mapping)
    elif encoding_method == "One-Hot Encoding":
        if unique_count <= max_onehot_unique:
            encoding_map[col] = "One-Hot Encoding"
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else:
            encoding_map[col] = "Frequency Encoding (High Cardinality fallback)"
            freq = safe_value_counts(safe_categorical_series(df[col]), normalize=True)
            df[col + '_freq'] = safe_categorical_series(df[col]).map(freq)
            df.drop(columns=[col], inplace=True)
    elif unique_count == 2:
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
        freq = safe_value_counts(safe_categorical_series(df[col]), normalize=True)
        df[col + '_freq'] = safe_categorical_series(df[col]).map(freq)
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
    "2) Why feature scaling should be done after train/test split during model training "
    "3) Encoding strategies (Ordinal vs Nominal vs High Cardinality). Keep it to 5 bullet points.",
    f"Skewed features handled: {len(high_skew)}. Encoding strategies applied: {len(encoding_map)}.",
)
st.markdown(f'<div class="ai-label">🤖 AI ANALYSIS</div><div class="ai-insight">{ai_step5}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# STEP 11 — Feature Selection (Variance Filter)
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 11 — Feature Selection (Variance Filter)", "⚙️")

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

ai_step12_var = ask_deepseek(
    "You are a senior ML engineer. Explain what feature engineering involves, "
    "why correlation analysis matters, and what zero-variance features are. "
    "Also explain the risks of highly correlated features (multicollinearity). "
    "5 bullet points, interview-ready.",
    f"Numeric features: {len(num_cols_fe)}. Zero-variance dropped: {len(low_var)}",
)
st.markdown(f'<div class="ai-label">🤖 AI EXPLANATION</div><div class="ai-insight">{ai_step12_var}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# STEP 11 — Feature Selection (Correlation Filter)
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 11 — Feature Selection (Correlation Filter)", "🔬")

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

ai_step12_corr = ask_deepseek(


    "You are a senior ML engineer. Explain feature selection techniques (filter, wrapper, embedded), "
    "why removing highly correlated features improves model performance, "
    "and mention Variance Inflation Factor (VIF). 5 bullet points, interview-ready.",
    f"Features before: {len(num_cols_fs)}. Dropped for high correlation (>0.95): {len(dropped_corr)}",
)
st.markdown(f'<div class="ai-label">🤖 AI EXPLANATION</div><div class="ai-insight">{ai_step12_corr}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# STEP 12 — Final Dataset Preparation
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 12 — Final Dataset Preparation (Train/Test Split)", "✂️")

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

ai_step13 = ask_deepseek(
    "You are a senior ML engineer. Explain train/test split best practices: "
    "why 80/20, stratified split, data leakage risks, and cross-validation. "
    "5 bullet points, interview-ready.",
    f"Split: {100*(1-test_size):.0f}/{100*test_size:.0f}. Train: {len(train_df)}, Test: {len(test_df)}",
)
st.markdown(f'<div class="ai-label">🤖 AI EXPLANATION</div><div class="ai-insight">{ai_step13}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# STEP 12 — Export cleaned dataset
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 12 — Export Cleaned Dataset", "💾")

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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats

# 1. Load
df = pd.read_csv("your_file.csv")
print("Shape:", df.shape)

# 1.1 Drop only primary ID column(s), not all high-unique columns
target_guess = df.columns[-1]  # keep likely supervised target safe
id_tokens = ("id", "uuid", "identifier", "case_no", "caseid", "record_no")
id_cols = []
for col in df.columns:
    if col == target_guess:
        continue
    s = df[col].dropna()
    if len(s) == 0:
        continue
    unique_ratio = s.nunique() / len(s)
    name_suggests_id = any(tok in col.lower() for tok in id_tokens)
    index_like = False
    if pd.api.types.is_integer_dtype(s):
        d = s.diff().dropna()
        index_like = len(d) > 0 and (d.abs() == 1).mean() >= 0.98
    if unique_ratio >= 0.995 and (name_suggests_id or index_like):
        id_cols.append(col)
if id_cols:
    df.drop(columns=id_cols, inplace=True)
    print(f"Dropped primary ID columns: {{id_cols}}")

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

# 5. Outliers — IQR cap
for col in df.select_dtypes(include="number").columns:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - {iqr_factor}*IQR, Q3 + {iqr_factor}*IQR)

# 6–7. Drop zero-variance & highly correlated
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=0.0)
vt.fit(df)
df = df[df.columns[vt.get_support()]]

corr = df.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
df.drop(columns=to_drop, inplace=True)

# 8. Train/Test split
train_df, test_df = train_test_split(df, test_size={test_size}, random_state=42)
print(f"Train: {{len(train_df)}} | Test: {{len(test_df)}}")

# 9. Save
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

st.success("✅ Pipeline completed through Step 12 as configured.")
st.stop()

# ─────────────────────────────────────────────────────────
# STEP 13 — End-to-End Model Training, Evaluation & Tuning
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
section("Step 13 — End-to-End Model Training, Evaluation & Tuning", "🤖")
st.caption("Uses only the cleaned dataset above and enforces split-first preprocessing to prevent leakage.")

detected_target = detect_target_column(df)
if detected_target is None or detected_target not in df.columns:
    st.info("🎯 No target selected/detected. Running unsupervised modeling pipeline (PCA + clustering).")

    X_unsup_raw = df.copy()
    X_unsup, unsup_summary = build_unsupervised_feature_matrix(X_unsup_raw)

    if X_unsup.shape[0] < 10 or X_unsup.shape[1] < 2:
        st.warning("⚠️ Unsupervised modeling needs at least 10 rows and 2 usable numeric features after encoding.")
    else:
        st.markdown(
            f"**Prepared feature matrix:** `{unsup_summary['rows']}` rows × "
            f"`{unsup_summary['features_after_encoding']}` features"
        )
        if unsup_summary["datetime_cols_converted"]:
            st.info(
                "ℹ️ Datetime columns converted to numeric timestamps: "
                + ", ".join(unsup_summary["datetime_cols_converted"])
            )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_unsup)

        max_components = min(10, X_scaled.shape[1], X_scaled.shape[0] - 1)
        pca_full = PCA(n_components=max_components, random_state=42)
        X_pca_full = pca_full.fit_transform(X_scaled)
        explained = pca_full.explained_variance_ratio_
        cum_explained = np.cumsum(explained)
        optimal_components = int(np.argmax(cum_explained >= 0.95) + 1) if np.any(cum_explained >= 0.95) else max_components
        optimal_components = max(2, min(optimal_components, max_components))

        st.markdown("#### PCA — Explained Variance")
        pca_var_df = pd.DataFrame(
            {
                "PC": [f"PC{i+1}" for i in range(len(explained))],
                "Explained Variance": explained,
                "Cumulative Variance": cum_explained,
            }
        )
        st.dataframe(pca_var_df, use_container_width=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(range(1, len(cum_explained) + 1), cum_explained, marker="o", color="#62b8ff")
        ax.axhline(0.95, linestyle="--", color="orange", linewidth=1)
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title("PCA Cumulative Explained Variance")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        X_pca_model = PCA(n_components=optimal_components, random_state=42).fit_transform(X_scaled)
        X_pca_2d = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

        cluster_rows = []
        best_labels = None
        best_model_name = None
        best_score = -1.0

        max_k = min(8, max(2, X_pca_model.shape[0] - 1))
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_pca_model)
            n_clusters = len(set(labels))
            if n_clusters > 1:
                sil = silhouette_score(X_pca_model, labels)
                cluster_rows.append({"Model": f"KMeans (k={k})", "Clusters": n_clusters, "Silhouette": sil})
                if sil > best_score:
                    best_score = sil
                    best_labels = labels
                    best_model_name = f"KMeans (k={k})"

        eps_val = st.slider("DBSCAN eps", 0.1, 3.0, 0.8, 0.1, key="unsup_dbscan_eps")
        min_samples_val = st.slider("DBSCAN min_samples", 3, 20, 5, 1, key="unsup_dbscan_min_samples")
        db = DBSCAN(eps=eps_val, min_samples=min_samples_val)
        db_labels = db.fit_predict(X_pca_model)
        db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        if db_clusters > 1 and np.unique(db_labels).size > 1:
            db_sil = silhouette_score(X_pca_model, db_labels)
            cluster_rows.append({"Model": "DBSCAN", "Clusters": db_clusters, "Silhouette": db_sil})
            if db_sil > best_score:
                best_score = db_sil
                best_labels = db_labels
                best_model_name = "DBSCAN"
        else:
            cluster_rows.append({"Model": "DBSCAN", "Clusters": db_clusters, "Silhouette": np.nan})

        cluster_df = pd.DataFrame(cluster_rows).sort_values("Silhouette", ascending=False, na_position="last")
        st.markdown("#### Unsupervised Model Comparison")
        st.dataframe(cluster_df, use_container_width=True)

        if best_labels is not None and best_model_name is not None:
            st.success(f"✅ Best clustering model: {best_model_name} (silhouette={best_score:.4f})")
            plot_df = pd.DataFrame(
                {
                    "PC1": X_pca_2d[:, 0],
                    "PC2": X_pca_2d[:, 1],
                    "Cluster": best_labels.astype(str),
                }
            )
            st.markdown("#### PCA (2D) Cluster Visualization")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Cluster", palette="tab10", ax=ax, alpha=0.8)
            ax.set_title(f"PCA Cluster View — {best_model_name}")
            ax.grid(alpha=0.25)
            st.pyplot(fig)
        else:
            st.warning("⚠️ Could not form stable clusters with current settings. Try adjusting DBSCAN controls.")
else:
    if st.session_state.get("selected_target") != detected_target:
        st.info(f"🎯 Auto-detected target column: `{detected_target}`")

    y = df[detected_target]
    X = df.drop(columns=[detected_target])

    if X.shape[1] == 0:
        st.warning("⚠️ No feature columns are available after target selection. Skipping modeling.")
    elif y.nunique(dropna=True) < 2:
        st.warning("⚠️ Target has fewer than 2 unique values. Skipping modeling.")
    else:
        detected_problem_type = infer_problem_type(y)
        y_notnull_mask = y.notna()
        dropped_target_nan = int((~y_notnull_mask).sum())
        if dropped_target_nan > 0:
            X = X.loc[y_notnull_mask].copy()
            y = y.loc[y_notnull_mask].copy()
            st.info(f"ℹ️ Dropped {dropped_target_nan} row(s) with missing target values before modeling.")

        st.markdown(
            f"**Target variable:** `{detected_target}` &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"**Detected problem type:** `{detected_problem_type.title()}`"
        )

        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
        scaler_name = "StandardScaler"
        ctrl_col1.metric("Feature scaler", "StandardScaler (train-fit only)")
        ctrl_col2.metric("Configured CV folds", f"{model_cv_folds_cfg}")
        ctrl_col3.metric("Configured tuning iterations", f"{model_tuning_iterations_cfg}")

        training_logs = []
        training_box = st.empty()

        def push_training_update(message: str, state: str = "running"):
            training_logs.append(message)
            render_training_status_box(training_box, training_logs, status=state)

        push_training_update("Initializing train-test split and train-safe missing handling.")

        split_kwargs = {"test_size": 0.20, "random_state": 42}
        if detected_problem_type == "classification":
            class_counts_all = y.value_counts()
            if class_counts_all.min() >= 2:
                split_kwargs["stratify"] = y
            else:
                st.warning("⚠️ At least one class has <2 samples, so random split is used instead of stratified split.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
        X_train_model, X_test_model, y_train_model, y_test_model, missing_summary = handle_missing_for_modeling(
            X_train, X_test, y_train, y_test, low_nan_row_threshold=0.01
        )

        if missing_summary["dropped_all_nan_cols"]:
            st.info(f"ℹ️ Dropped all-NaN feature column(s): {', '.join(missing_summary['dropped_all_nan_cols'])}")
        if missing_summary["dropped_train_rows"] > 0:
            st.info(f"ℹ️ Dropped {missing_summary['dropped_train_rows']} low-NaN train row(s).")
        if missing_summary["dropped_test_rows"] > 0:
            st.info(f"ℹ️ Dropped {missing_summary['dropped_test_rows']} low-NaN test row(s).")
        if missing_summary["imputed_numeric_cols"] > 0 or missing_summary["imputed_categorical_cols"] > 0:
            st.info(
                f"ℹ️ Imputed missing values using TRAIN statistics — numeric(mean): "
                f"{missing_summary['imputed_numeric_cols']}, categorical(mode): {missing_summary['imputed_categorical_cols']}."
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("Train samples", f"{len(X_train_model):,}")
        m2.metric("Test samples", f"{len(X_test_model):,}")
        m3.metric("Split ratio", "80/20")
        push_training_update(
            f"Data ready: train={len(X_train_model)} samples, test={len(X_test_model)} samples."
        )

        use_smote = False
        if detected_problem_type == "classification":
            train_class_dist = y_train_model.value_counts().rename_axis("Class").reset_index(name="Count")
            st.markdown("**Training class distribution:**")
            st.dataframe(train_class_dist, use_container_width=True)

            min_count = y_train_model.value_counts().min()
            max_count = y_train_model.value_counts().max()
            imbalance_ratio = min_count / max_count if max_count else 1.0
            use_smote = imbalance_ratio < 0.60 and min_count >= 2

            if use_smote and IMBLEARN_AVAILABLE:
                st.success(
                    "✅ Imbalance detected. SMOTE will be applied only inside the model-training pipeline on TRAIN data."
                )
            elif use_smote and not IMBLEARN_AVAILABLE:
                st.warning("⚠️ Imbalance detected, but `imbalanced-learn` is not installed, so SMOTE is skipped.")
            else:
                st.info("ℹ️ Class imbalance is not significant; SMOTE is not needed.")
            push_training_update(
                "Class imbalance check completed. "
                + ("SMOTE applied on train split." if use_smote and IMBLEARN_AVAILABLE else "Proceeding without SMOTE.")
            )

        requested_cv_folds = int(model_cv_folds_cfg)
        max_tuning_iterations = int(model_tuning_iterations_cfg)
        cv_cap = len(X_train_model)
        if detected_problem_type == "classification":
            cv_cap = min(cv_cap, int(y_train_model.value_counts().min()))
        cv_enabled = cv_cap >= 2
        cv_folds = min(requested_cv_folds, cv_cap) if cv_enabled else 0
        if cv_enabled and cv_folds != requested_cv_folds:
            st.info(
                f"ℹ️ Requested CV={requested_cv_folds}, but data supports up to {cv_cap} folds. "
                f"Using CV={cv_folds}."
            )
        if not cv_enabled:
            st.warning("⚠️ Not enough samples to run cross-validation (requires at least 2 folds).")
            push_training_update("Cross-validation disabled due to low sample support.")
        else:
            push_training_update(f"Cross-validation configured with {cv_folds} folds.")

        model_registry = get_model_registry(detected_problem_type)
        baseline_rows = []
        baseline_models = {}
        push_training_update("Training baseline models for leaderboard.")

        for model_name, cfg in model_registry.items():
            pipeline = build_model_pipeline(
                clone(cfg["estimator"]),
                scaler_name=scaler_name,
                use_scaler=cfg["use_scaler"],
                use_smote=use_smote and detected_problem_type == "classification",
            )
            pipeline.fit(X_train_model, y_train_model)
            baseline_models[model_name] = pipeline
            metrics, _ = evaluate_predictions(detected_problem_type, pipeline, X_test_model, y_test_model)
            baseline_rows.append({"Model": model_name, **metrics})

        baseline_df = pd.DataFrame(baseline_rows)
        if detected_problem_type == "classification":
            baseline_df["RankScore"] = (
                baseline_df["F1-score"].fillna(0)
                + 0.20 * baseline_df["ROC-AUC"].fillna(0)
                + 0.10 * baseline_df["Accuracy"].fillna(0)
            )
            baseline_df = baseline_df.sort_values(["RankScore", "F1-score"], ascending=False).reset_index(drop=True)
        else:
            baseline_df["RankScore"] = -baseline_df["RMSE"]
            baseline_df = baseline_df.sort_values(["RMSE", "MAE"], ascending=True).reset_index(drop=True)

        st.markdown("#### Model Comparison (Baseline on Test Split)")
        st.dataframe(baseline_df, use_container_width=True)

        top_models = baseline_df.head(3)["Model"].tolist()
        st.markdown(f"**Top 3 models:** {', '.join(top_models)}")
        push_training_update(f"Baseline comparison done. Top models: {', '.join(top_models)}.")

        cv_rows = []
        cv_scoring = "f1_weighted" if detected_problem_type == "classification" else "neg_root_mean_squared_error"
        if cv_enabled:
            push_training_update("Running cross-validation on top 3 models.")
            for model_name in top_models:
                cfg = model_registry[model_name]
                cv_pipeline = build_model_pipeline(
                    clone(cfg["estimator"]),
                    scaler_name=scaler_name,
                    use_scaler=cfg["use_scaler"],
                    use_smote=use_smote and detected_problem_type == "classification",
                )
                cv_scores = cross_val_score(
                    cv_pipeline, X_train_model, y_train_model, cv=cv_folds, scoring=cv_scoring, n_jobs=-1
                )
                if detected_problem_type == "classification":
                    cv_rows.append(
                        {
                            "Model": model_name,
                            "CV folds": cv_folds,
                            "CV Score (f1_weighted mean)": cv_scores.mean(),
                            "CV Std": cv_scores.std(),
                        }
                    )
                else:
                    cv_rows.append(
                        {
                            "Model": model_name,
                            "CV folds": cv_folds,
                            "CV Score (RMSE mean)": -cv_scores.mean(),
                            "CV Std": cv_scores.std(),
                        }
                    )
            push_training_update("Cross-validation completed for top models.")
        else:
            for model_name in top_models:
                cv_rows.append(
                    {
                        "Model": model_name,
                        "CV folds": "N/A",
                        "CV Score (f1_weighted mean)" if detected_problem_type == "classification" else "CV Score (RMSE mean)": np.nan,
                        "CV Std": np.nan,
                    }
                )

        cv_df = pd.DataFrame(cv_rows)
        st.markdown("#### Cross-Validation (Top 3 Models)")
        st.dataframe(cv_df, use_container_width=True)

        tuned_models = {}
        tuning_rows = []
        tuning_epoch_history = []
        tuning_scoring = "f1_weighted" if detected_problem_type == "classification" else "neg_root_mean_squared_error"
        planned_tuning_epochs = sum(
            min(max_tuning_iterations, get_search_space_size(model_registry[m]["params"]))
            for m in top_models
            if model_registry[m]["params"] and cv_enabled
        )
        completed_tuning_epochs = 0

        push_training_update(f"Starting hyperparameter tuning (max iterations per model: {max_tuning_iterations}).")
        if planned_tuning_epochs > 0:
            tuning_progress = st.progress(
                0.0,
                text=f"Hyperparameter tuning epochs: 0/{planned_tuning_epochs}",
            )
        else:
            tuning_progress = None
        for idx, model_name in enumerate(top_models, start=1):
            cfg = model_registry[model_name]
            tune_pipeline = build_model_pipeline(
                clone(cfg["estimator"]),
                scaler_name=scaler_name,
                use_scaler=cfg["use_scaler"],
                use_smote=use_smote and detected_problem_type == "classification",
            )
            param_dist = cfg["params"]
            push_training_update(f"Tuning {model_name} ({idx}/{len(top_models)}).")
            if not param_dist or not cv_enabled:
                tune_pipeline.fit(X_train_model, y_train_model)
                tuned_models[model_name] = {
                    "estimator": tune_pipeline,
                    "best_score": np.nan,
                    "best_params": {},
                }
                tuning_rows.append(
                    {
                        "Model": model_name,
                        "Training Score": tune_pipeline.score(X_train_model, y_train_model),
                        "Iterations Used": 0,
                        "Validation Score (CV mean)": np.nan,
                        "Best Params": "{}",
                    }
                )
                push_training_update(f"{model_name} has no tunable search space or CV is disabled; fitted directly.")
                continue

            n_iter = min(max_tuning_iterations, get_search_space_size(param_dist))
            best_score = -np.inf
            best_params = {}
            random_states = [42, 42 + idx]
            sampled_params = []
            for seed in random_states:
                sampled_params.extend(list(ParameterSampler(param_dist, n_iter=n_iter, random_state=seed)))
                if len(sampled_params) >= n_iter:
                    break
            sampled_params = sampled_params[:n_iter]

            for epoch_idx, params in enumerate(sampled_params, start=1):
                candidate_pipeline = build_model_pipeline(
                    clone(cfg["estimator"]),
                    scaler_name=scaler_name,
                    use_scaler=cfg["use_scaler"],
                    use_smote=use_smote and detected_problem_type == "classification",
                )
                candidate_pipeline.set_params(**params)
                cv_scores = cross_val_score(
                    candidate_pipeline,
                    X_train_model,
                    y_train_model,
                    cv=cv_folds,
                    scoring=tuning_scoring,
                    n_jobs=-1,
                )
                epoch_score = float(cv_scores.mean())
                epoch_loss = score_to_loss(detected_problem_type, epoch_score)
                is_best_epoch = epoch_score > best_score
                if epoch_score > best_score:
                    best_score = epoch_score
                    best_params = params

                tuning_epoch_history.append(
                    {
                        "Model": model_name,
                        "Epoch": epoch_idx,
                        "CV Score": epoch_score,
                        "Validation Loss": epoch_loss,
                        "Best So Far": "Yes" if is_best_epoch else "No",
                    }
                )

                completed_tuning_epochs += 1
                if tuning_progress is not None:
                    tuning_progress.progress(
                        completed_tuning_epochs / planned_tuning_epochs,
                        text=(
                            f"Hyperparameter tuning epochs: {completed_tuning_epochs}/{planned_tuning_epochs} "
                            f"(current: {model_name} epoch {epoch_idx}/{n_iter})"
                        ),
                    )
                push_training_update(
                    f"{model_name} epoch {epoch_idx}/{n_iter} completed. "
                    f"Current CV: {epoch_score:.4f} | Current loss: {epoch_loss:.4f} | Best CV: {best_score:.4f}"
                )

            best_pipeline = build_model_pipeline(
                clone(cfg["estimator"]),
                scaler_name=scaler_name,
                use_scaler=cfg["use_scaler"],
                use_smote=use_smote and detected_problem_type == "classification",
            )
            if best_params:
                best_pipeline.set_params(**best_params)
            best_pipeline.fit(X_train_model, y_train_model)
            tuned_models[model_name] = {
                "estimator": best_pipeline,
                "best_score": best_score,
                "best_params": best_params,
            }
            tuning_rows.append(
                {
                    "Model": model_name,
                    "Training Score": best_pipeline.score(X_train_model, y_train_model),
                    "Iterations Used": n_iter,
                    "Validation Score (CV mean)": best_score,
                    "Best Params": json.dumps(best_params),
                }
            )
            push_training_update(
                f"{model_name} tuned with {n_iter} epochs. Best CV score: {best_score:.4f}."
            )
        if tuning_progress is not None:
            tuning_progress.progress(
                1.0,
                text=f"Hyperparameter tuning epochs: {planned_tuning_epochs}/{planned_tuning_epochs} (complete)",
            )

        tuning_df = pd.DataFrame(tuning_rows)
        st.markdown("#### Hyperparameter Tuning Results (Top 3)")
        st.dataframe(tuning_df, use_container_width=True)

        st.markdown("#### Epoch-wise Loss Tracking")
        if tuning_epoch_history:
            tuning_epoch_df = pd.DataFrame(tuning_epoch_history)
            fig, ax = plt.subplots(figsize=(9, 4.8))
            sns.lineplot(
                data=tuning_epoch_df,
                x="Epoch",
                y="Validation Loss",
                hue="Model",
                marker="o",
                ax=ax,
            )
            ax.set_title("Validation Loss per Epoch (Hyperparameter Tuning)")
            ax.set_ylabel("Loss (1 - F1 weighted)" if detected_problem_type == "classification" else "Loss (RMSE)")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            with st.expander("View epoch-by-epoch scores and loss"):
                st.dataframe(tuning_epoch_df, use_container_width=True)
        else:
            st.info("ℹ️ No epoch history available (no tunable search space or CV disabled).")

        final_rows = []
        push_training_update("Evaluating tuned models on the holdout test set.")
        for model_name, tuned in tuned_models.items():
            tuned_estimator = tuned["estimator"]
            metrics, y_pred = evaluate_predictions(detected_problem_type, tuned_estimator, X_test_model, y_test_model)
            final_rows.append({"Model": model_name, **metrics})

        final_df = pd.DataFrame(final_rows)
        if detected_problem_type == "classification":
            final_df["RankScore"] = (
                final_df["F1-score"].fillna(0)
                + 0.20 * final_df["ROC-AUC"].fillna(0)
                + 0.10 * final_df["Accuracy"].fillna(0)
            )
            final_df = final_df.sort_values(["RankScore", "F1-score"], ascending=False).reset_index(drop=True)
        else:
            final_df["RankScore"] = -final_df["RMSE"]
            final_df = final_df.sort_values(["RMSE", "MAE"], ascending=True).reset_index(drop=True)

        st.markdown("#### Final Evaluation on Test Data (Tuned Models)")
        st.dataframe(final_df, use_container_width=True)

        best_model_name = final_df.iloc[0]["Model"]
        best_model = tuned_models[best_model_name]["estimator"]
        best_metrics, best_preds = evaluate_predictions(detected_problem_type, best_model, X_test_model, y_test_model)
        push_training_update(f"Best model selected: {best_model_name}.", state="complete")

        st.markdown("#### Training vs Validation Performance")
        best_tuning_row = tuning_df[tuning_df["Model"] == best_model_name].iloc[0]
        perf_col1, perf_col2 = st.columns(2)
        perf_col1.metric("Training Score", f"{best_tuning_row['Training Score']:.4f}")
        if pd.isna(best_tuning_row["Validation Score (CV mean)"]):
            perf_col2.metric("Validation Score (CV mean)", "N/A")
        else:
            perf_col2.metric("Validation Score (CV mean)", f"{best_tuning_row['Validation Score (CV mean)']:.4f}")

        lc_scoring = "f1_weighted" if detected_problem_type == "classification" else "neg_root_mean_squared_error"
        if cv_enabled:
            train_sizes, train_scores, val_scores = learning_curve(
                best_model,
                X_train_model,
                y_train_model,
                cv=cv_folds,
                scoring=lc_scoring,
                train_sizes=np.linspace(0.1, 1.0, 5),
                n_jobs=-1,
            )
            train_curve = train_scores.mean(axis=1)
            val_curve = val_scores.mean(axis=1)
            if detected_problem_type == "regression":
                train_curve = -train_curve
                val_curve = -val_curve

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(train_sizes, train_curve, marker="o", label="Training")
            ax.plot(train_sizes, val_curve, marker="o", label="Validation")
            ax.set_title(f"Learning Curve — {best_model_name}")
            ax.set_xlabel("Training Samples")
            ax.set_ylabel("Score" if detected_problem_type == "classification" else "RMSE")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        if detected_problem_type == "classification":
            st.markdown("#### Confusion Matrix (Best Model)")
            cm = confusion_matrix(y_test_model, best_preds)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix — {best_model_name}")
            st.pyplot(fig)
            st.markdown("#### Classification Report")
            st.code(classification_report(y_test_model, best_preds), language="text")
        else:
            st.markdown("#### Residual Plot (Best Model)")
            residuals = y_test_model - best_preds
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.scatter(best_preds, residuals, alpha=0.65)
            ax.axhline(0, color="red", linestyle="--", linewidth=1)
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Residuals")
            ax.set_title(f"Residual Plot — {best_model_name}")
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        st.markdown("#### Feature Importance")
        model_core = best_model.named_steps["model"] if hasattr(best_model, "named_steps") else best_model
        feature_names = X.columns.tolist()
        importances = None
        if hasattr(model_core, "feature_importances_"):
            importances = model_core.feature_importances_
        elif hasattr(model_core, "coef_"):
            coef = np.array(model_core.coef_)
            importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)

        if importances is not None and len(importances) == len(feature_names):
            fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            fi_df = fi_df.sort_values("Importance", ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.barplot(data=fi_df, x="Importance", y="Feature", ax=ax, palette="viridis")
            ax.set_title(f"Top Feature Importances — {best_model_name}")
            st.pyplot(fig)
        else:
            st.info("ℹ️ Feature importance is not available for this model type.")

        st.markdown("#### Final Output")
        st.markdown(
            f"""
**Best model:** `{best_model_name}`  
**Best hyperparameters:** `{tuned_models[best_model_name]['best_params']}`  
**Final test metrics:** `{best_metrics}`  
**Selection rationale:** Highest test performance among tuned models with stronger train/validation consistency.
"""
        )

# ─────────────────────────────────────────────────────────
# STEP 14 & 15 — Visualization Standards + Final Summary
# ─────────────────────────────────────────────────────────
st.markdown("---")
section("Step 14 — Visualization Standards", "🖼️")
st.markdown(
    "This run includes professional visualizations such as histogram, KDE, box plot, violin plot, "
    "count/bar charts, scatter plots, pair plot, and heatmap where applicable."
)
section("Step 15 — Final Summary", "🎯")

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
5. Feature Selection: Removed {cols_removed} low-variance/highly-correlated features
6. Train/Test Split: {int((1-test_size)*100)}% train / {int(test_size*100)}% test

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
