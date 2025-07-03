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

col1, col2 = st.columns(2)

with col1:
    txn_type = st.selectbox(
        'Transaction Type',
        ('PAYMENT', 'CASH_OUT', 'TRANSFER', 'CASH_IN', 'DEBIT')
    )
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
            # 1. Create a dictionary from the user's raw input
            input_data = {
                'txn_type': txn_type,
                'txn_amount': txn_amount,
                'sender_balance_before': sender_balance_before,
                'sender_balance_after': sender_balance_after,
                'receiver_balance_before': receiver_balance_before,
                'receiver_balance_after': receiver_balance_after,
            }

            # 2. Convert to a DataFrame
            input_df = pd.DataFrame([input_data])
            
            # --- THIS IS THE CRUCIAL FIX ---
            # 3. Engineer the features just like in the notebook
            input_df['sender_balance_error'] = input_df['sender_balance_after'] + input_df['txn_amount'] - input_df['sender_balance_before']
            input_df['receiver_balance_error'] = input_df['receiver_balance_after'] - input_df['txn_amount'] - input_df['receiver_balance_before']
            # -------------------------------

            # 4. Make Prediction
            # The model now receives the data in the exact format it was trained on
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
                st.write("Data Sent to Model (after feature engineering):")
                st.dataframe(input_df)

# --- How It Works Section ---
st.markdown("---")
with st.expander("How does this app work?"):
    st.markdown("""
    1.  **Input Data**: You provide the details of a financial transaction.
    2.  **Feature Engineering**: The app first calculates the `_balance_error` features, which are highly predictive of fraud.
    3.  **Data Transformation**: This complete DataFrame (with engineered features) is then fed into the saved pipeline. The pipeline automatically handles scaling and one-hot encoding.
    4.  **Prediction**: The fully processed data is passed to the trained XGBoost model, which outputs a fraud probability score.
    """)
