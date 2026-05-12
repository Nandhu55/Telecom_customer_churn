import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

model_path = os.path.join(os.path.dirname(__file__), "churn_model.pkl")
model = pickle.load(open(model_path, "rb"))

st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }

    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background-color: #4F46E5;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border: none;
    }

    .stButton>button:hover {
        background-color: #6366F1;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #A1A1AA;
        margin-bottom: 30px;
    }

    .stSelectbox label,
    .stNumberInput label {
        font-weight: 600;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

def main():

    st.markdown('<div class="title">📱 Customer Churn Prediction</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="subtitle">Predict whether a telecom customer is likely to churn</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

    with col2:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=1000.0
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict Churn"):

        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
            'tenure': [tenure],
            'Contract': [contract],
            'PaperlessBilling': [1 if paperless_billing == "Yes" else 0],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'Partner': [0],
            'Dependents': [0],
            'PhoneService': [0],
            'MultipleLines': ['No'],
            'InternetService': ['No'],
            'OnlineSecurity': ['No'],
            'OnlineBackup': ['No'],
            'DeviceProtection': ['No'],
            'TechSupport': ['No'],
            'StreamingTV': ['No'],
            'StreamingMovies': ['No']
        })

        input_data = pd.get_dummies(input_data)

        input_data = input_data.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.markdown("---")

        if prediction == 1:
            st.error(
                f"⚠️ Customer is likely to churn\n\nProbability: {probability:.2%}"
            )
        else:
            st.success(
                f"✅ Customer is unlikely to churn\n\nProbability: {(1 - probability):.2%}"
            )

if __name__ == "__main__":
    main()