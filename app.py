import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('model.pkl', 'rb'))

st.title("🧠 Brain Stroke Prediction App")
st.write("Fill in the details below to predict stroke risk.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100)
hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes"])

# Encode categorical values
gender = 1 if gender == "Male" else 0
ever_married = 1 if ever_married == "Yes" else 0
residence_type = 1 if residence_type == "Urban" else 0
work_type_map = {"Private": 2, "Self-employed": 3, "Govt_job": 0}
work_type = work_type_map[work_type]
smoking_map = {"formerly smoked": 0, "never smoked": 1, "smokes": 2}
smoking_status = smoking_map[smoking_status]

# Predict
if st.button("Predict"):
    input_data = np.array([[gender, age, hypertension, heart_disease,
                            ever_married, work_type, residence_type,
                            avg_glucose_level, bmi, smoking_status]])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("⚠️ High risk of Stroke detected!")
    else:
        st.success("✅ Low risk of Stroke. Stay healthy!")