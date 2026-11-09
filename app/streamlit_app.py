import streamlit as st, pandas as pd, joblib
from src.score import score_frame
st.title("Credit Card Fraud Risk Scoring")
uploaded=st.file_uploader("Upload transactions CSV",type="csv")
if uploaded:
    data=pd.read_csv(uploaded); artifact=joblib.load("models/logistic_balanced.joblib"); result=score_frame(data,artifact); st.metric("Review threshold",f"{artifact['threshold']:.2f}"); st.bar_chart(result["risk_tier"].value_counts()); st.dataframe(result.sort_values("fraud_probability",ascending=False))
