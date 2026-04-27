import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and encoder
model = joblib.load('random_forest_regressor.pkl')
le = joblib.load('label_encoder.pkl')

st.title("Salary Prediction App")
st.write("Enter details to predict the estimated salary.")

# Create input fields based on the training features
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD", "High School"])
job_title = st.text_input("Job Title", "Software Engineer")
experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=5.0)

if st.button("Predict Salary"):
    # Prepare the input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Education Level': [education],
        'Job Title': [job_title],
        'Years of Experience': [experience]
    })

    # Apply the same Label Encoding logic
    # Note: In a production app, we should handle unseen labels more robustly
    for col in ['Gender', 'Education Level', 'Job Title']:
        try:
            # Try to transform using the loaded encoder
            # (Simplified: using the encoder's classes to match)
            input_data[col] = le.fit_transform(input_data[col]) 
        except:
            input_data[col] = 0

    # Make prediction
    prediction = model.predict(input_data)
    
    st.success(f"The estimated salary is: INR{prediction[0]:,.2f}")
