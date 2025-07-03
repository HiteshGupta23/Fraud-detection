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

# --- Define the exact column order the model was trained on ---
# This is the most critical part of the fix.
MODEL_COLUMN_ORDER = [
    'txn_type',
    'txn_amount',
    'sender_balance_before',
    'sender_balance_after',
    'receiver_balance_before',
    'receiver_balance_after',
    'sender_balance_error',
    'receiver_balance_error'
]

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

col1, col2 = st.columns(2)

with col1:
    txn_type = st.selectbox('Transaction Type', ('PAYMENT', 'CASH_OUT', 'TRANSFER', 'CASH_IN', 'DEBIT'))
    sender_balance_before = st.number_input('Sender Balance Before', min_value=0.0, format="%.2f")
    receiver_balance_before = st.number_input('Receiver Balance Before', min_value=0.0, format="%.2f")

with col2:
    txn_amount = st.number_input('Transaction Amount', min_value=0.0, format="%.2f")
    sender_balance_after = st.number_input('Sender Balance After', min_value=0.0, format="%.2f")
    receiver_balance_after = st.number_input('Receiver Balance After', min_value=0.0, format="%.2f")

st.divider()

# --- Prediction Logic ---
_, col_button, _ = st.columns([2, 1, 2])
with col_button:
    if st.button("Analyze Transaction", use_container_width=True):
        if model is not None:
            # 1. Create a DataFrame from the user's raw input
            input_df = pd.DataFrame([{
                'txn_type': txn_type,
                'txn_amount': txn_amount,
                'sender_balance_before': sender_balance_before,
                'sender_balance_after': sender_balance_after,
                'receiver_balance_before': receiver_balance_before,
                'receiver_balance_after': receiver_balance_after,
            }])
            
            # 2. Engineer the new features
            input_df['sender_balance_error'] = input_df['sender_balance_after'] + input_df['txn_amount'] - input_df['sender_balance_before']
            input_df['receiver_balance_error'] = input_df['receiver_balance_after'] - input_df['txn_amount'] - input_df['receiver_balance_before']
            
            # 3. *** THE CRUCIAL FIX: Enforce the correct column order ***
            input_df = input_df[MODEL_COLUMN_ORDER]

            # 4. Make Prediction
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            # --- Display Results ---
            st.subheader("Fraud Analysis Result")
            # ... (rest of the display logic is the same) ...
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
                st.write("Data Sent to Model (Reordered & Engineered):")
                st.dataframe(input_df)

# --- How It Works Section ---
# ... (rest of the app is the same) ...
st.markdown("---")
with st.expander("How does this app work?"):
    st.markdown("""
    1.  **Input Data**: You provide the details of a financial transaction.
    2.  **Feature Engineering**: The app first calculates the `_balance_error` features.
    3.  **Data Structuring**: The app then **reorders the columns** to precisely match the structure the model was trained on. This is a critical step.
    4.  **Transformation & Prediction**: This complete DataFrame is fed into the saved pipeline, which handles scaling, encoding, and finally, prediction.
    """)
