# 🚀 AI-Powered Automated EDA, Feature Engineering & ML Pipeline Agent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-ML_App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![OpenRouter](https://img.shields.io/badge/OpenRouter-AI-green)
![License](https://img.shields.io/badge/License-MIT-purple)

</div>

---

# 📌 Project Overview

## 🧠 Intelligent AI-Driven Machine Learning Preprocessing System

This project is an advanced AI-powered machine learning preprocessing and exploratory data analysis platform that automates the complete data preparation workflow using:

- Artificial Intelligence
- Automated EDA
- Feature Engineering
- Feature Selection
- Data Cleaning
- Model Training
- Hyperparameter Tuning
- Explainable AI

The system is designed to intelligently analyze uploaded datasets and automatically determine:

- Whether the problem is:
  - Classification
  - Regression
  - Unsupervised Learning
- Which preprocessing techniques should be applied
- Which feature engineering strategies are suitable
- Which machine learning workflow should be executed

The project combines deterministic preprocessing pipelines with AI-generated reasoning and explanations using OpenRouter APIs.

---

# 🎯 Problem Statement

In real-world machine learning projects, data preprocessing and exploratory data analysis consume nearly 70–80% of the development lifecycle.

Traditional preprocessing workflows suffer from several limitations:

- Manual EDA consumes significant time
- Choosing preprocessing techniques requires domain expertise
- Feature engineering is difficult for beginners
- Identifying the ML problem type is not always straightforward
- Data cleaning pipelines are repetitive
- Lack of explainability in preprocessing decisions
- Reproducibility challenges

This project solves these problems by creating an intelligent automated preprocessing ecosystem capable of:

✅ Understanding datasets automatically  
✅ Identifying ML problem type  
✅ Performing dynamic preprocessing  
✅ Executing AI-assisted EDA  
✅ Applying adaptive feature engineering  
✅ Training models automatically  
✅ Explaining every major step using AI

---

# 💡 Core Idea of the Project

The user uploads a dataset and provides:

- Dataset description
- Problem statement
- Business objective

The AI engine analyzes:

- Dataset structure
- Feature distributions
- Target column characteristics
- Data types
- Missing values
- Cardinality
- Statistical properties

Then the system automatically decides:

| Dataset Condition         | AI Decision                    |
| ------------------------- | ------------------------------ |
| Categorical Target        | Classification Pipeline        |
| Continuous Numeric Target | Regression Pipeline            |
| No Target Column          | Unsupervised Learning Pipeline |

Based on this decision, the application dynamically modifies:

- EDA workflow
- Feature engineering logic
- Encoding strategy
- Scaling strategy
- Model selection
- Evaluation metrics

---

# 🧠 AI-Powered Dynamic Workflow

# 📊 Intelligent Problem Type Detection

The AI automatically classifies the dataset into:

## ✅ Classification

Detected when:

- Target column contains categories/classes
- Binary labels
- Multi-class labels

Examples:

- Spam Detection
- Disease Prediction
- Loan Approval
- Fraud Detection

### Workflow Activated

- Class distribution analysis
- Imbalance detection
- SMOTE recommendation
- Label encoding
- Classification algorithms
- Precision / Recall / F1 evaluation

---

## ✅ Regression

Detected when:

- Target variable is continuous numerical

Examples:

- House Price Prediction
- Sales Forecasting
- Stock Prediction

### Workflow Activated

- Outlier sensitivity analysis
- Distribution analysis
- Skewness correction
- Scaling strategies
- Regression models
- RMSE / MAE / R² evaluation

---

## ✅ Unsupervised Learning

Detected when:

- No target column exists
- User selects clustering/unsupervised mode

Examples:

- Customer Segmentation
- Pattern Discovery
- Market Basket Analysis

### Workflow Activated

- PCA
- Clustering preparation
- Dimensionality reduction
- Correlation structure analysis
- Anomaly detection

---

# ⚡ End-to-End Pipeline Architecture

```text
User Uploads CSV Dataset
            ↓
Dataset Description + Problem Statement
            ↓
AI Dataset Understanding
            ↓
Automatic Problem Type Detection
            ↓
Dynamic EDA Pipeline Generation
            ↓
Missing Value Handling
            ↓
Duplicate Detection & Removal
            ↓
Outlier Detection & Treatment
            ↓
Feature Engineering
            ↓
Feature Selection
            ↓
Encoding & Transformation
            ↓
Train/Test Split
            ↓
Model Training
            ↓
Cross Validation
            ↓
Hyperparameter Tuning
            ↓
Evaluation & Visualization
            ↓
Export Cleaned Dataset + Python Script
```

---

# 🔥 Major Features

# 📊 Automated Exploratory Data Analysis (EDA)

The application automatically performs:

## Univariate Analysis

- Histograms
- KDE plots
- Count plots
- Box plots
- Skewness analysis

## Bivariate Analysis

- Scatter plots
- Feature-target relationships
- Correlation studies

## Multivariate Analysis

- Correlation heatmaps
- Pair plots
- Feature interaction analysis

---

# 🧹 Intelligent Data Cleaning

## Missing Value Handling

Supported strategies:

- Mean
- Median
- Mode
- Drop rows
- AI-recommended strategy

## Duplicate Handling

- Duplicate detection
- Automatic removal

## Outlier Treatment

Methods:

- IQR Method
- Z-Score Method
- Capping
- Removal

---

# ⚙️ AI-Based Feature Engineering

The system dynamically applies:

## Encoding Strategies

- Label Encoding
- One-Hot Encoding
- Frequency Encoding

## Feature Transformations

- Log transformation
- Skewness correction
- Scaling preparation

## Feature Selection

- Correlation threshold removal
- Zero variance feature removal
- Redundant feature elimination

---

# 🤖 Automated Machine Learning

After preprocessing:

## Classification Models

- Logistic Regression
- Random Forest
- Decision Tree
- XGBoost

## Regression Models

- Linear Regression
- Random Forest Regressor
- Gradient Boosting
- XGBoost Regressor

## Unsupervised Models

- K-Means
- DBSCAN
- PCA

---

# 🧠 OpenRouter AI Integration

This project integrates OpenRouter APIs to provide:

- AI dataset explanations
- AI preprocessing recommendations
- Interview-style reasoning
- Dynamic pipeline configuration
- Intelligent workflow generation

---

# 🏗️ Technical Architecture

## Frontend

- Streamlit

## Backend

- Python

## Data Processing

- Pandas
- NumPy

## Machine Learning

- Scikit-Learn
- SciPy

## Visualization

- Matplotlib
- Seaborn

## AI Integration

- OpenRouter
- OpenAI SDK

---

# 📂 Project Structure

```text
preprocessing_agent/
│
├── preprocessing_agent.py
├── app.py
├── requirements.txt
├── README.md
├── .env
└── datasets/
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone <your-repository-url>
cd preprocessing_agent
```

---

## Create Virtual Environment

```bash
python -m venv .venv
```

---

## Activate Environment

### Linux / Mac

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Configuration

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

---

# ▶️ Run the Application

```bash
streamlit run preprocessing_agent.py
```

---

# 📈 Output Generated

The system provides:

✅ Cleaned Dataset
✅ Downloadable CSV
✅ Generated Python Script
✅ Visual Diagnostics
✅ AI Explanations
✅ Model Evaluation Reports
✅ Feature Engineering Insights
✅ Correlation Reports

---

# 🧠 Why This Project Is Unique

Unlike traditional preprocessing systems, this project:

✅ Dynamically changes workflow based on ML problem type
✅ Uses AI to explain preprocessing decisions
✅ Automates feature engineering
✅ Provides interview-level ML explanations
✅ Combines deterministic ML with Generative AI
✅ Supports classification, regression, and unsupervised learning
✅ Generates reproducible preprocessing scripts

---

# 🎓 Educational Value

This project is highly valuable for:

- Data Science Students
- Machine Learning Engineers
- AI Researchers
- Beginners learning EDA
- Interview Preparation
- Academic Projects
- Portfolio Development

---

# 📌 Future Improvements

- AutoML integration
- Deep Learning support
- Time series preprocessing
- NLP preprocessing pipelines
- LLM-powered feature synthesis
- Real-time dashboard analytics
- Cloud deployment support

---

# 🏆 Resume Description

Developed an AI-powered automated machine learning preprocessing and exploratory data analysis platform capable of dynamically identifying machine learning problem types and performing adaptive EDA, feature engineering, preprocessing, model training, and AI-generated explanations using Streamlit, Scikit-Learn, and OpenRouter APIs.

---

# 📜 License

MIT License

---

# ⭐ Support

If you like this project:

⭐ Star the repository
🍴 Fork the project
🧠 Contribute improvements

---

# 👨‍💻 Author

J Anand

---
