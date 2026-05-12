# Streamlit App: Salary Prediction Model with Complete Analysis

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


st.set_page_config(layout="wide")
st.title("💼 Employee Salary Prediction Dashboard")

# ── Load & cache dataset ──────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    try:
        df = pd.read_csv("Salary_Data.csv")
    except FileNotFoundError:
        st.error("❌ Could not find Salary_Data.csv. Please place it in the same folder as app.py.")
        st.stop()

    df.dropna(inplace=True)

    # Reduce rare job titles
    job_title_counts = df['Job Title'].value_counts()
    rare_titles = job_title_counts[job_title_counts <= 25].index
    df['Job Title'] = df['Job Title'].apply(lambda x: 'Others' if x in rare_titles else x)
    df['Original Job Title'] = df['Job Title']

    # Standardize education labels
    edu_map = {"Bachelor's Degree": "Bachelor's", "Master's Degree": "Master's", "phD": "PhD"}
    df['Education Level'] = df['Education Level'].replace(edu_map)

    # Encode gender
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])

    # Encode education
    edu_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    df['Education Level'] = df['Education Level'].map(edu_mapping)

    # One-hot encode Job Title
    df = pd.get_dummies(df, columns=['Job Title'], drop_first=True)

    return df, le, edu_mapping


@st.cache_data
def train_models(df_encoded):
    y = df_encoded['Salary']
    X = df_encoded.drop(columns=['Salary', 'Original Job Title'])

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, min_samples_split=2, random_state=0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    model_results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        model_results[name] = {
            'Model': model,
            'R² Score': round(model.score(x_test, y_test), 4),
            'MAE': round(mean_absolute_error(y_test, y_pred), 2),
            'RMSE': round(mean_squared_error(y_test, y_pred) ** 0.5, 2)
        }

    return model_results, X, x_test, y_test


# ── Run pipeline ──────────────────────────────────────────────────────────────
df, le, edu_mapping = load_and_preprocess()
model_results, X, x_test, y_test = train_models(df)

# ── Step 1: Dataset Preview ───────────────────────────────────────────────────
st.subheader("Step 1: Dataset Preview")
st.write(df[['Age', 'Gender', 'Education Level', 'Years of Experience', 'Salary']].head())
st.caption(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ── Step 2: Model Evaluation ──────────────────────────────────────────────────
st.subheader("Step 2: Model Evaluation")
metrics_df = pd.DataFrame(model_results).T.drop(columns='Model')
st.dataframe(metrics_df, width='stretch')

# ── Step 3: Visual Analysis ───────────────────────────────────────────────────
st.subheader("Step 3: Visual Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("### Gender Distribution")
    fig, ax = plt.subplots()
    gender_labels = df.copy()
    gender_labels['Gender'] = gender_labels['Gender'].map(dict(enumerate(le.classes_)))
    sns.countplot(x='Gender', data=gender_labels, ax=ax, hue='Gender', palette='pastel', legend=False)
    st.pyplot(fig)

with col2:
    st.write("### Education Level Distribution")
    fig, ax = plt.subplots()
    edu_labels = df.copy()
    edu_labels['Education Level'] = edu_labels['Education Level'].map(
        {v: k for k, v in edu_mapping.items()}
    )
    sns.countplot(x='Education Level', data=edu_labels, ax=ax, hue='Education Level',
              palette='pastel', legend=False, order=["High School", "Bachelor's", "Master's", "PhD"])
    st.pyplot(fig)

st.write("### Top 10 Highest Paying Job Titles")
top_jobs = df.groupby('Original Job Title')['Salary'].mean().nlargest(10)
fig, ax = plt.subplots(figsize=(10, 4))
top_jobs.sort_values().plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel("Average Salary (USD)")
ax.set_title("Top 10 Jobs by Average Salary")
st.pyplot(fig)

st.write("### Feature Importance (Random Forest)")
rfr = model_results['Random Forest']['Model']
importance = rfr.feature_importances_
indices = np.argsort(importance)[::-1][:10]
important_features = X.columns[indices]
fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(important_features[::-1], importance[indices][::-1], color='coral')
ax.set_title("Top 10 Most Important Features")
st.pyplot(fig)

# ── Step 4: Predict Salary ────────────────────────────────────────────────────
st.subheader("Step 4: Predict Your Salary")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 65, 30)
        gender = st.selectbox("Gender", le.classes_)
        education = st.selectbox("Education Level", list(edu_mapping.keys()))
    with col2:
        experience = st.slider("Years of Experience", 0, 40, 5)
        job_options = sorted(
            [col.replace('Job Title_', '') for col in X.columns if 'Job Title_' in col] + ['Others']
        )
        job_input = st.selectbox("Job Title", job_options)

    predict_btn = st.form_submit_button("🔍 Predict Salary")

if predict_btn:
    row = {
        'Age': age,
        'Gender': le.transform([gender])[0],
        'Education Level': edu_mapping[education],
        'Years of Experience': experience
    }
    for col in X.columns:
        if col.startswith('Job Title_'):
            row[col] = 1 if job_input == col.replace('Job Title_', '') else 0

    input_df = pd.DataFrame([row])[X.columns]

    st.write("### 📊 Predicted Salary")
    res_col1, res_col2, res_col3 = st.columns(3)
    cols = [res_col1, res_col2, res_col3]
    for i, (name, result) in enumerate(model_results.items()):
        pred_salary = result['Model'].predict(input_df)[0]
        cols[i].metric(label=name, value=f"${pred_salary:,.0f}")
        
    # ── GenAI Explanation ─────────────────────────────────────────────────────
    st.write("### 🤖 AI Explanation")

    best_model_name = max(model_results, key=lambda x: model_results[x]['R² Score'])
    best_pred = model_results[best_model_name]['Model'].predict(input_df)[0]

    prompt = f"""
    A salary prediction model estimated a salary of {best_pred:,.0f} for an employee with:
    - Age: {age}
    - Gender: {gender}
    - Education: {education}
    - Years of Experience: {experience}
    - Job Title: {job_input}

    In 3-4 sentences, explain in plain English why this salary makes sense.
    Reference which factors (experience, education, job title) likely drove this result
    and how it compares to typical market compensation. Be professional and concise.
    """

    with st.spinner("Generating explanation..."):
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        gemini = genai.GenerativeModel("gemini-1.5-flash")
        response = gemini.generate_content(prompt)
        st.info(response.text)









