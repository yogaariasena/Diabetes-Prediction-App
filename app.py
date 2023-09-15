import streamlit as st
import joblib
import pandas as pd

# Load the model
pipeline = joblib.load('diabetes_prediction_model.joblib')

st.title("Diabetes Prediction App")

st.sidebar.header("User Input")

# Create input fields for user to enter data
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
age = st.sidebar.slider("Age", 0.08, 80.0, 25.0)
hypertension = st.sidebar.selectbox("Hypertension", ['No', 'Yes'])
heart_disease = st.sidebar.selectbox("Heart Disease", ['No', 'Yes'])
smoking_history = st.sidebar.selectbox("Smoking History", ['No Info', 'current', 'former', 'ever', 'not current'])
bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0)
HbA1c_level = st.sidebar.number_input("HbA1c Level", 3.5, 9.0, 6.0)
blood_glucose_level = st.sidebar.slider("Blood Glucose Level", 80, 300, 120)

# Create a DataFrame from the user's input
data = {
    'gender': [gender],
    'age': [age],
    'hypertension': [1 if hypertension == 'Yes' else 0],
    'heart_disease': [1 if heart_disease == 'Yes' else 0],
    'smoking_history': [smoking_history],
    'bmi': [bmi],
    'HbA1c_level': [HbA1c_level],
    'blood_glucose_level': [blood_glucose_level]
}

user_input = pd.DataFrame(data)

# Make a prediction
prediction = pipeline.predict(user_input)

st.subheader("Prediction")

if prediction[0] == 1:
    st.write("The model predicts that the individual has diabetes.")
else:
    st.write("The model predicts that the individual does not have diabetes.")
