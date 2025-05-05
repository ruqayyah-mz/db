# diabetes_rf_streamlit.py
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

# Streamlit UI
st.title("Diabetes Progression Predictor")
st.write("Input the following parameters to predict the disease progression (quantitative measure)")

# Create input widgets for all features
user_input = {}
for feature in diabetes.feature_names:
    val = st.number_input(f"Enter {feature}", value=float(X[feature].mean()), step=0.01)
    user_input[feature] = val

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted disease progression value: {prediction:.2f}")
