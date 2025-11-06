import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ----------------------------------------------------
# ✅ Load Saved Model
# ----------------------------------------------------
model = pickle.load(open("model.pkl", "rb"))

# ----------------------------------------------------
# ✅ App Title
# ----------------------------------------------------
st.title("Loan Payback Prediction App")
st.write("Fill the details below to predict whether the borrower will pay back the loan.")

# ----------------------------------------------------
# ✅ User Input Fields
# ----------------------------------------------------

# Numerical Inputs
annual_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
dti = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, value=20.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=10000)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.0)

# Categorical Inputs
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD", "Other"])
employment_status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed", "Student"])
loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Education", "Medical", "Home", "Business", "Other"])

# ----------------------------------------------------
# ✅ Convert Input to DataFrame (for Pipeline Models)
# ----------------------------------------------------
input_df = pd.DataFrame({
    "annual_income": [annual_income],
    "debt_to_income_ratio": [dti],
    "credit_score": [credit_score],
    "loan_amount": [loan_amount],
    "interest_rate": [interest_rate],
    "gender": [gender],
    "marital_status": [marital_status],
    "education_level": [education_level],
    "employment_status": [employment_status],
    "loan_purpose": [loan_purpose]
})

# ----------------------------------------------------
# ✅ Predict Button
# ----------------------------------------------------
if st.button("Predict Loan Payback"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("✅ Loan is **likely to be paid back!**")
    else:
        st.error("❌ Loan is **likely NOT to be paid back.**")
