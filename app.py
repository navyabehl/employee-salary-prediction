# Streamlit App: Salary Prediction Model with Complete Analysis
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(layout="wide")
st.title("Salary Prediction Dashboard")

# Load dataset
try:
    df = pd.read_csv("C:/Users/NAVYA/Downloads/Salary_Data.csv")
except:
    st.error("❌ Could not load Salary_Data.csv. Please check the path.")
    st.stop()

st.subheader("Step 1: Dataset Preview")
st.write(df.head())

# Step 2: Preprocessing
st.subheader("Step 2: Data Cleaning and Preprocessing")
df.dropna(inplace=True)

# Reduce job titles
job_title_counts = df['Job Title'].value_counts()
rare_titles = job_title_counts[job_title_counts <= 25].index
df['Job Title'] = df['Job Title'].apply(lambda x: 'Others' if x in rare_titles else x)
df['Original Job Title'] = df['Job Title']

# Standardize education labels
edu_map = {"Bachelor's Degree": "Bachelor's", "Master's Degree": "Master's", "phD": "PhD"}
df['Education Level'] = df['Education Level'].replace(edu_map)

# Encode categorical columns
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
edu_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
df['Education Level'] = df['Education Level'].map(edu_mapping)

# One-hot encoding for Job Title
df = pd.get_dummies(df, columns=['Job Title'], drop_first=True)

# Target and Features
y = df['Salary']
X = df.drop(columns=['Salary', 'Original Job Title'])

# Train-test split
if len(X) < 2:
    st.error("Not enough data after preprocessing. Check your dataset.")
    st.stop()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 3: Train Models
st.subheader("Step 3: Training Models")
model_results = {}
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, min_samples_split=2, random_state=0),
    'Random Forest': RandomForestRegressor(n_estimators=20)
}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    model_results[name] = {
        'Model': model,
        'R2 Score': model.score(x_test, y_test),
        'MSE': mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': mean_squared_error(y_test, y_pred) ** 0.5
    }

# Step 4: Show Metrics
st.subheader("Step 4: Model Evaluation")
metrics_df = pd.DataFrame(model_results).T.drop(columns='Model')
st.dataframe(metrics_df)

# Step 5: Plots
st.subheader("Step 5: Visual Analysis")

col1, col2 = st.columns(2)
with col1:
    st.write("### Gender Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Gender', data=df.replace({'Gender': dict(enumerate(le.classes_))}), ax=ax)
    st.pyplot(fig)
with col2:
    st.write("### Education Level Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Education Level', data=df.replace({'Education Level': {v: k for k, v in edu_mapping.items()}}), ax=ax)
    st.pyplot(fig)

st.write("### Top 10 Highest Paying Jobs")
top_jobs = df.groupby('Original Job Title')['Salary'].mean().nlargest(10)
fig, ax = plt.subplots()
top_jobs.plot(kind='bar', ax=ax)
ax.set_ylabel("Average Salary")
ax.set_title("Top 10 Jobs")
st.pyplot(fig)

st.write("### Feature Importance (Random Forest)")
rfr = model_results['Random Forest']['Model']
importance = rfr.feature_importances_
indices = np.argsort(importance)[::-1][:10]
important_features = X.columns[indices]
fig, ax = plt.subplots()
ax.barh(important_features[::-1], importance[indices][::-1])
ax.set_title("Top 10 Important Features")
st.pyplot(fig)

# Step 6: User Prediction
st.subheader("Step 6: Predict Salary from Your Input")
with st.form("prediction_form"):
    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", le.classes_)
    education = st.selectbox("Education Level", list(edu_mapping.keys()))
    experience = st.slider("Years of Experience", 0, 40, 5)
    job_input = st.selectbox("Job Title", sorted([col.replace('Job Title_', '') for col in X.columns if 'Job Title_' in col] + ['Others']))
    predict_btn = st.form_submit_button("Predict")

if predict_btn:
    # Build input row
    row = {
        'Age': age,
        'Gender': le.transform([gender])[0],
        'Education Level': edu_mapping[education],
        'Years of Experience': experience
    }
    for col in X.columns:
        if col.startswith('Job Title_'):
            row[col] = 1 if job_input == col.replace('Job Title_', '') else 0
    input_df = pd.DataFrame([row])[X.columns]  # Ensure order
    st.write("### Predicted Salary from Each Model")
    for name, result in model_results.items():
        pred_salary = result['Model'].predict(input_df)[0]
        st.write(f"{name}: ${pred_salary:,.2f}")
