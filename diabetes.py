# diabetes_rf_app.py
import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Friendly display names for features
feature_labels = {
    'age': 'Age (standardized)',
    'sex': 'Sex (1 = male, 0 = female)',
    'bmi': 'Body Mass Index',
    'bp': 'Average Blood Pressure',
    's1': 'Total Serum Cholesterol',
    's2': 'Low-Density Lipoproteins (LDL)',
    's3': 'High-Density Lipoproteins (HDL)',
    's4': 'Total Cholesterol / HDL Ratio',
    's5': 'Blood Sugar Level',
    's6': 'Insulin Level'
}

# Streamlit UI
st.title("🩺 Diabetes Progression Predictor")
st.write("Provide the following inputs to predict disease progression score.")

# Collect user inputs
user_input = {}

for feature in diabetes.feature_names:
    label = feature_labels.get(feature, feature)
    
    if feature == 'sex':
        sex_input = st.selectbox(label, options=['Male', 'Female'])
        user_input[feature] = 1.0 if sex_input == 'Male' else 0.0
    else:
        default_val = float(X[feature].mean())
        user_input[feature] = st.number_input(label, value=default_val, step=0.01)

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted disease progression value: {prediction:.2f}")
