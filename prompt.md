You are a senior Data Scientist. Perform a COMPLETE, AUTOMATED Exploratory Data Analysis (EDA) and Data Preprocessing pipeline on the given dataset.

The solution must intelligently adapt to ANY dataset and automatically detect whether it is SUPERVISED or UNSUPERVISED.

---

🔹 1. Problem Type Detection (CRITICAL)

- Check if a target variable exists:
  - If YES → Treat as SUPERVISED learning
  - If NO → Treat as UNSUPERVISED learning

- If target is not explicitly mentioned, infer the most likely target and justify
- Identify:
  - Classification (categorical target)
  - Regression (continuous target)
  - Clustering / pattern discovery (no target)

---

🔹 2. Data Overview

- Dataset shape (rows, columns)
- Column names and data types
- Show first 5 rows
- Summary statistics (numerical + categorical)
- Memory usage

---

🔹 3. Data Cleaning

- Missing values:
  - Show percentage per column
  - Handle appropriately (drop / mean / median / mode / advanced)

- Duplicate rows:
  - Detect and remove

- Incorrect data types:
  - Fix (datetime, categorical, numeric conversions)

- Handle inconsistent or noisy values

---

🔹 4. Column Categorization (AUTO-DETECT)

- Numerical:
  - Continuous Features (float values, wide range)
  - Discrete Features (integer count-based values)

- Categorical:
  - Nominal
  - Ordinal

- Datetime features

---

🔹 5. Univariate Analysis (DETAILED)

========================
🔢 Numerical Features
=====================

🔸 Continuous Features:

- Histogram
- KDE plot
- Box plot (outlier detection)
- Violin plot (distribution shape)
- Skewness & distribution analysis

🔸 Discrete Features:

- Value counts
- Bar plot / Count plot
- Box plot (if needed)

👉 Compare distribution differences between discrete vs continuous clearly

---

========================
🔤 Categorical Features
=======================

- Count plot
- Bar plot
- Pie chart (proportion visualization)
- Rare category detection

---

🔹 6. CONDITIONAL ANALYSIS BASED ON DATA TYPE

========================
📊 IF SUPERVISED DATASET
========================

🔸 Target Analysis:

- Classification:
  - Class distribution (count plot, pie chart)
  - Detect imbalance

- Regression:
  - Target distribution (histogram, KDE)
  - Skewness and outliers

🔸 Feature vs Target Analysis:

🔹 Continuous vs Target:

- Scatter plot
- Correlation analysis

🔹 Discrete vs Target:

- Bar plot (mean target per value)
- Box plot

🔹 Categorical vs Target:

- Box plot
- Violin plot
- Bar plot

🔸 Correlation Analysis:

- Heatmap (numerical features)
- Detect multicollinearity

🔸 Feature Importance Thinking:

- Identify strong predictors

🔸 Data Leakage Check:

- Detect leakage features

👉 Focus: Relationship between FEATURES (X) and TARGET (Y)

---

========================
📊 IF UNSUPERVISED DATASET
==========================

🔸 Feature Relationship Analysis:

- Scatter plots
- Pair plot (if feasible)
- Correlation heatmap

🔸 Dimensionality Reduction:

- Apply PCA:
  - Explained variance ratio
  - 2D visualization

- Apply t-SNE (if needed)

🔸 Clustering Readiness:

- Elbow method
- Silhouette score

🔸 Clustering:

- K-Means
- Hierarchical clustering
- DBSCAN

🔸 Cluster Visualization:

- Plot clusters using PCA/t-SNE

🔸 Cluster Interpretation:

- Analyze cluster-wise feature distributions

👉 Focus: Pattern discovery & grouping

---

🔹 7. Multivariate Analysis

- Pairplot (if feasible)
- Feature interactions
- Multicollinearity detection

---

🔹 8. Outlier Detection & Handling

- IQR method
- Z-score method
- Decide:
  - Remove
  - Cap
  - Retain

---

🔹 9. Feature Engineering

- Create new meaningful features
- Datetime extraction (year, month, weekday)
- Binning
- Interaction features

---

🔹 10. Encoding

- Label Encoding (ordinal)
- One-Hot Encoding (nominal)
- Handle high cardinality

---

🔹 11. Feature Selection

- Correlation filtering
- Variance threshold
- Model-based importance

---

🔹 12. Final Dataset Preparation

- Train-test split (ONLY for supervised)
- Show final dataset shape
- Avoid data leakage

---

🔹 13. Visualization Standards
Use professional plots:

- Histogram
- KDE plot
- Box plot
- Violin plot
- Count plot
- Bar plot
- Pie chart
- Scatter plot
- Pair plot
- Heatmap

---

🔹 14. Final Summary

- Key insights
- Important features or clusters
- Data issues and fixes
- Preprocessing decisions

---

IMPORTANT RULES:

- Automatically skip irrelevant steps
- Do NOT assume column names
- Handle edge cases (missing data, imbalance, high cardinality)
- Keep output clean and structured

---

Dataset:
<PASTE YOUR DATASET / FILE / DESCRIPTION HERE>

ValueError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/eda_feature_engineering_autoanted_with_ai_detailed_explanation/app.py", line 2260, in <module>
np.fill_diagonal(corr_abs.values, np.nan)
~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.14/site-packages/numpy/lib/\_index_tricks_impl.py", line 923, in fill_diagonal
a.flat[:end:step] = val
~~~~~~^^^^^^^^^^^
