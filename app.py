import streamlit as st
import pandas as pd
import numpy as np
import joblib
from utils import PreProcer, columns

model = joblib.load("xgbpipe.joblib")

st.title("Survived or Not")

name = st.text_input("name:")
sex = st.selectbox("sex:", ["male", "female"])
pclass = st.selectbox("class? :", [1,2,3])
ticket = st.text_input("ticket:")
sibsp = st.slider("chosse sibsp:", 0,10)
age = st.slider("chosse age:", 0,100)
parch = st.slider("chosse parch:" , 0,10)
fare = st.number_input("fare:")
cabin = st.text_input("Cabin:", "CS2")
embarked = st.radio("chosse embarked", ["S","C","Q"])

def predict():
    row = np.array([name,sex,pclass,ticket,sibsp,age,parch,fare,cabin,embarked])
    X = pd.DataFrame([row], columns=columns)

    prediction = model.predict(X)
    
    if prediction[0] == 1:
        st.success("Passenger Survived! ")
    else:
        st.error("Passenger Died! ")

trigger = st.button("Predict", on_click=predict)