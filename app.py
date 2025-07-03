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
This application uses a sophisticated XGBoost model to predict fraudulent transactions. 
Enter the transaction details below to get an instant fraud risk assessment.
""")

# --- Main Page Inputs in Columns ---
st.divider()
st.subheader("Enter Transaction Details to Analyze 💳")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    txn_type = st.selectbox(
        'Transaction Type',
        ('PAYMENT', 'CASH_OUT', 'TRANSFER', 'CASH_IN', 'DEBIT'),
        key='txn_type' # Add a unique key for each widget
    )
    sender_balance_before = st.number_input('Sender Balance Before', min_value=0.0, format="%.2f", key='sender_before')
    receiver_balance_before = st.number_input('Receiver Balance Before', min_value=0.0, format="%.2f", key='receiver_before')

with col2:
    txn_amount = st.number_input('Transaction Amount', min_value=0.0, format="%.2f", key='amount')
    sender_balance_after = st.number_input('Sender Balance After', min_value=0.0, format="%.2f", key='sender_after')
    receiver_balance_after = st.number_input('Receiver Balance After', min_value=0.0, format="%.2f", key='receiver_after')

st.divider()

# --- Prediction Logic ---
# The button is now on the main page, centered
_, col_button, _ = st.columns([2, 1, 2])
with col_button:
    if st.button("Analyze Transaction", use_container_width=True):
        if model is not None:
            # Create a dictionary from the user's input
            input_data = {
                'txn_type': txn_type,
                'txn_amount': txn_amount,
                'sender_balance_before': sender_balance_before,
                'sender_balance_after': sender_balance_after,
                'receiver_balance_before': receiver_balance_before,
                'receiver_balance_after': receiver_balance_after,
            }
            # Convert to DataFrame for the model
            input_df = pd.DataFrame([input_data])

            # Make Prediction
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
    2.  **Data Transformation**: The raw data is fed directly into a saved pipeline object which automatically performs all necessary preprocessing steps.
    3.  **Prediction**: The processed data is then passed to the trained XGBoost model within the pipeline, which outputs a fraud probability score.
    4.  **Result**: The app interprets the score to give a clear, human-readable result.
    """)
