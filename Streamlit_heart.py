import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('XGBoost.pkl')

# Define feature options
cp_options = {
    1: 'Typical angina (1)',
    2: 'Atypical angina (2)',
    3: 'Non-anginal pain (3)',
    4: 'Asymptomatic (4)'
}

restecg_options = {
    0: 'Normal (0)',
    1: 'ST-T wave abnormality (1)',
    2: 'Left ventricular hypertrophy (2)'
}

slope_options = {
    1: 'Upsloping (1)',
    2: 'Flat (2)',
    3: 'Downsloping (3)'
}

thal_options = {
    1: 'Normal (1)',
    2: 'Fixed defect (2)',
    3: 'Reversible defect (3)'
}

# Define feature names
feature_names = [
    "Age", "Sex", "Chest Pain Type", "Resting Blood Pressure", "Serum Cholesterol",
    "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate", "Exercise Induced Angina",
    "ST Depression", "Slope", "Number of Vessels", "Thal"
]

# Streamlit user interface
st.title("Heart Disease Predictor")

# age: numerical input
age = st.number_input("Age:", min_value=1, max_value=120, value=50)

# sex: categorical selection
sex = st.selectbox("Sex (0=Female, 1=Male):", options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')

# cp: categorical selection
cp = st.selectbox("Chest pain type:", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])

# trestbps: numerical input
trestbps = st.number_input("Resting blood pressure (trestbps):", min_value=50, max_value=200, value=120)

# chol: numerical input
chol = st.number_input("Serum cholesterol in mg/dl (chol):", min_value=100, max_value=600, value=200)

# fbs: categorical selection
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs):", options=[0, 1], format_func=lambda x: 'False (0)' if x == 0 else 'True (1)')

# restecg: categorical selection
restecg = st.selectbox("Resting electrocardiographic results:", options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x])

# thalach: numerical input
thalach = st.number_input("Maximum heart rate achieved (thalach):", min_value=50, max_value=250, value=150)

# exang: categorical selection
exang = st.selectbox("Exercise induced angina (exang):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# oldpeak: numerical input
oldpeak = st.number_input("ST depression induced by exercise relative to rest (oldpeak):", min_value=0.0, max_value=10.0, value=1.0)

# slope: categorical selection
slope = st.selectbox("Slope of the peak exercise ST segment (slope):", options=list(slope_options.keys()), format_func=lambda x: slope_options[x])

# ca: numerical input
ca = st.number_input("Number of major vessels colored by fluoroscopy (ca):", min_value=0, max_value=4, value=0)

# thal: categorical selection
thal = st.selectbox("Thal (thal):", options=list(thal_options.keys()), format_func=lambda x: thal_options[x])

# Process inputs and make predictions
feature_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of heart disease. "
            f"The model predicts that your probability of having heart disease is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of heart disease. "
            f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your heart health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")