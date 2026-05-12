# 💼 Employee Salary Prediction — ML + GenAI Dashboard

An end-to-end machine learning project that predicts employee salaries based on demographic and professional attributes, with an integrated **Generative AI explanation layer** powered by Google Gemini. Built as part of the **AICTE–IBM Edunet AI/ML Internship (2025)**.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-red?logo=streamlit)](https://navyabehl-employee-salary-prediction.streamlit.app/)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-GenAI-orange?logo=google)](https://aistudio.google.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🚀 Live Demo

**[▶ Try the app here](https://navyabehl-employee-salary-prediction.streamlit.app/)**

---

## 📌 Project Overview

This project analyzes a dataset of 10,000+ employee records to build and compare predictive models for compensation benchmarking. The interactive Streamlit dashboard allows users to input their profile and receive real-time salary predictions across three ML models, along with a **plain-English AI explanation** of what drives their predicted compensation — powered by Google Gemini 2.0 Flash.

**Key Result:** Random Forest model achieved **R² = 0.97**, reducing prediction error margins by 35% over baseline.

---

## 🎯 Features

- **Exploratory Data Analysis** — Gender distribution, education levels, top-paying job titles
- **Feature Engineering** — Rare job title grouping, education level encoding, one-hot encoding
- **Multi-Model Comparison** — Linear Regression, Decision Tree, and Random Forest evaluated side-by-side
- **Interactive Prediction** — Input age, gender, education, experience, and job title to get instant salary estimates
- **Feature Importance** — Visual breakdown of which factors most influence salary predictions
- **GenAI Explanation Layer** — Google Gemini 2.0 Flash generates a natural language explanation of each prediction, improving model interpretability for non-technical users

---

## 🗂️ Repository Structure

```
employee-salary-prediction-ml/
│
├── app.py                            # Streamlit web application
├── salary_prediction_analysis.ipynb  # Full EDA and model development notebook
├── project_presentation.pptx         # Project summary presentation
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| ML Library | scikit-learn |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Web App | Streamlit |
| Generative AI | Google Gemini 2.0 Flash |

---

## 📊 Model Performance

| Model | R² Score | MAE | RMSE |
|---|---|---|---|
| Linear Regression | ~0.89 | 15,753.36 | 21,824.99 |
| Decision Tree | ~0.94 | 7,646.74 | 12,940.26 |
| **Random Forest** | **~0.97** | **3,430.20** | **8,990.93** |

> Random Forest was selected as the primary prediction model due to its superior accuracy and robustness to overfitting.

---

## 🤖 GenAI Integration

After generating salary predictions, the app calls the **Google Gemini 2.0 Flash** API to produce a concise, plain-English explanation of the predicted salary. This addresses a key challenge in ML deployment — **model explainability** — by translating numerical outputs into human-readable insights for non-technical stakeholders.

Example output:
> *"With 10 years of experience as a Data Scientist and a Master's degree, a salary of $120,000 aligns well with market benchmarks. Experience is the strongest driver here, contributing significantly more than education level alone..."*

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/navyabehl/employee-salary-prediction-ml.git
cd employee-salary-prediction-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your dataset
# Place Salary_Data.csv in the project root directory

# 4. Set up your Gemini API key
# Create a .streamlit/secrets.toml file with:
# GEMINI_API_KEY = "your-key-here"

# 5. Launch the app
streamlit run app.py
```

---

## 📁 Dataset

The model was trained on a publicly available salary dataset containing 10,000+ records with the following features:

- `Age` — Employee age
- `Gender` — Male / Female
- `Education Level` — High School, Bachelor's, Master's, PhD
- `Job Title` — 50+ roles (rare titles grouped as 'Others')
- `Years of Experience` — 0–40 years
- `Salary` — Target variable (in USD)

> **Note:** The dataset file (`Salary_Data.csv`) is not included in this repository. You can find a similar dataset on [Kaggle](https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer).

---

## 🔍 Key Insights

- **Years of Experience** is the strongest predictor of salary, followed by **Job Title** and **Education Level**
- PhD holders earn significantly more on average, especially in technical and managerial roles
- Rare/niche job titles were consolidated to prevent overfitting on low-frequency categories

---

## 👩‍💻 Author

**Navya Behl**
M.Sc. Economics | Dr. B.R. Ambedkar School of Economics University, Bengaluru

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/navyabehl)
[![GitHub](https://img.shields.io/badge/GitHub-navyabehl-black?logo=github)](https://github.com/navyabehl)

---

## 🏷️ Tags

`machine-learning` `salary-prediction` `streamlit` `random-forest` `scikit-learn` `python` `data-science` `HR-analytics` `compensation-benchmarking` `generative-ai` `google-gemini` `model-explainability`
