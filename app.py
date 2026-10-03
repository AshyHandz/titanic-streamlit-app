import streamlit as st
import pandas as pd
import joblib

st.title("Titanic Survival Predictor")
st.write("Input passenger details below to predict survival probability.")

#loading the trained model pipeline
model = joblib.load('titanic_model_pipeline.pkl')

#this is the input form
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Female", "Male"])
age = st.slider("Age", 1, 90, 30)
sibsp = st.number_input("Siblings / Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents / Children Aboard", 0, 10, 0)
fare = st.number_input("Ticket Fare ($)", 0.0, 500.0, 50.0)
embarked = st.selectbox("Embarked", ["Cherbourg", "Queenstown", "Southampton"])

#Map string back to numeric encoder (0=male, 1=female)
sex_encoded = 1 if sex=="Female" else 0
embarked_encoded = 1 if embarked=="Cherbourg" else 0

if st.button("Predict Survival"):
  input_df = pd.DataFrame([{
      'Pclass': pclass,
      'Sex': sex_encoded,
      'Age': age,
      'SibSp': sibsp,
      'Parch': parch,
      'Fare': fare,
      'Embarked': embarked_encoded
  }])

pred = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]

if pred == 1:
  st.success(f"Prediction: **Survived** ({prob * 100:.1f}% probability)")
  st.balloons()
else:
  st.error(f"Prediction: **Did Not Survive** ({(1 - prob)* 100:.1f}% probability)")
  st.snow()
#
