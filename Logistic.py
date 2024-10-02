# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the pre-trained model
model = pickle.load(open("Lipika_model.pkl", "rb"))

# Define a function for prediction
def predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    # Preprocess user input
    if Sex == "male":
        Sex = 1
    else:
        Sex = 0

    Embarked_dict = {"C": 0, "Q": 1, "S": 2}
    Embarked = Embarked_dict[Embarked]

    # Create an array for prediction
    input_features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    
    # Predict the survival probability
    prediction = model.predict(input_features)
    return "Survived" if prediction == 1 else "Did not Survive"

# Create the Streamlit web app interface
st.title("Titanic Survival Prediction")
st.write("Enter the passenger details to predict survival:")

# Input fields for passenger details
Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 100, 30)
SibSp = st.slider("Number of Siblings/Spouses Aboard", 0, 8, 0)
Parch = st.slider("Number of Parents/Children Aboard", 0, 6, 0)
Fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=50.0)
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Predict button
if st.button("Predict"):
    result = predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
    st.success(f"Prediction: {result}")
