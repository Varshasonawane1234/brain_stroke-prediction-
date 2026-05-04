import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def train_model():
    df = pd.read_csv('brain_stroke.csv')
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
    df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})
    df['work_type'] = df['work_type'].map({'Private': 2, 'Self-employed': 3, 'Govt_job': 0})
    df['smoking_status'] = df['smoking_status'].map({'formerly smoked': 0, 'never smoked': 1, 'smokes': 2})
    df = df.dropna()
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = train_model()

st.title("🧠 Brain Stroke Prediction App")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100)
hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job"])
Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes"])

gender = 1 if gender == "Male" else 0
ever_married = 1 if ever_married == "Yes" else 0
Residence_type = 1 if Residence_type == "Urban" else 0
work_type = {"Private": 2, "Self-employed": 3, "Govt_job": 0}[work_type]
smoking_status = {"formerly smoked": 0, "never smoked": 1, "smokes": 2}[smoking_status]

if st.button("Predict"):
    input_data = np.array([[gender, age, hypertension, heart_disease,
                            ever_married, work_type, Residence_type,
                            avg_glucose_level, bmi, smoking_status]])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("⚠️ High risk of Stroke detected!")
    else:
        st.success("✅ Low risk of Stroke. Stay healthy!")
