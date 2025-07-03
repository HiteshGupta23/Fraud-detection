import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Real-Time Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# --- Caching the Model ---
@st.cache_resource
def load_model():
    """Load the pre-trained XGBoost pipeline."""
    try:
        # The saved pipeline already includes the ColumnTransformer and the XGBoost model
        model = joblib.load('fraud_detection_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'fraud_detection_model.pkl' is in the root directory.")
        return None

# Load the model
model = load_model()

# --- Application Header ---
st.title("🛡️ Real-Time Transaction Fraud Detection")
st.markdown("""
This application uses a sophisticated XGBoost model to predict fraudulent transactions in real-time. 
Input the transaction details in the sidebar to get an instant fraud risk assessment.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("Transaction Details 💳")

# Define the input fields in the sidebar
txn_type = st.sidebar.selectbox(
    'Transaction Type',
    ('PAYMENT', 'CASH_OUT', 'TRANSFER', 'CASH_IN', 'DEBIT')
)
txn_amount = st.sidebar.number_input('Transaction Amount', min_value=0.0, format="%.2f")
sender_balance_before = st.sidebar.number_input('Sender Balance Before', min_value=0.0, format="%.2f")
sender_balance_after = st.sidebar.number_input('Sender Balance After', min_value=0.0, format="%.2f")
receiver_balance_before = st.sidebar.number_input('Receiver Balance Before', min_value=0.0, format="%.2f")
receiver_balance_after = st.sidebar.number_input('Receiver Balance After', min_value=0.0, format="%.2f")

# --- Prediction Logic ---
if st.sidebar.button("Analyze Transaction"):
    if model is not None:
        # 1. Create a dictionary from the user's input.
        # This dictionary should have the *original* raw feature names.
        input_data = {
            'txn_type': txn_type,
            'txn_amount': txn_amount,
            'sender_balance_before': sender_balance_before,
            'sender_balance_after': sender_balance_after,
            'receiver_balance_before': receiver_balance_before,
            'receiver_balance_after': receiver_balance_after,
        }

        # 2. Convert the dictionary to a DataFrame.
        # The model's internal ColumnTransformer will handle the rest.
        input_df = pd.DataFrame([input_data])
        
        # *** The manual feature engineering and one-hot encoding is NO LONGER NEEDED here ***
        # The saved pipeline will do it all automatically.

        # 3. Make Prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # --- Display Results ---
        st.subheader("Fraud Analysis Result")
        
        if prediction[0] == 1:
            st.error("High Risk: This transaction is likely FRAUDULENT!", icon="🚨")
        else:
            st.success("Low Risk: This transaction appears to be LEGITIMATE.", icon="✅")
        
        fraud_probability = prediction_proba[0][1]
        st.metric(label="Fraud Probability Score", value=f"{fraud_probability:.2%}")
        st.progress(fraud_probability)
        
        with st.expander("See Detailed Analysis"):
            st.write("Prediction Probabilities:")
            st.write(f"- Legitimate: {prediction_proba[0][0]:.2%}")
            st.write(f"- Fraudulent: {prediction_proba[0][1]:.2%}")
            st.write("---")
            st.write("Raw Data Sent to Model:")
            st.dataframe(input_df)

# --- How It Works Section ---
st.markdown("---")
with st.expander("How does this app work?"):
    st.markdown("""
    This app leverages a machine learning pipeline built with `scikit-learn` and `XGBoost`.
    1.  **Input Data**: You provide the details of a financial transaction.
    2.  **Data Transformation**: The raw data is fed directly into a saved pipeline object. This pipeline automatically performs all necessary preprocessing steps (like one-hot encoding categorical features) that were learned from the original training data.
    3.  **Prediction**: The processed data is then passed to the trained XGBoost model within the pipeline, which outputs a probability score of the transaction being fraudulent.
    4.  **Result**: The app interprets the score to give a clear, human-readable result.
    """)
