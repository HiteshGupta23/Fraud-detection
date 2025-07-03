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
# Use st.cache_resource to load the model only once, which improves performance.
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
        # 1. Create a dictionary from the user's input
        input_data = {
            'txn_amount': txn_amount,
            'sender_balance_before': sender_balance_before,
            'sender_balance_after': sender_balance_after,
            'receiver_balance_before': receiver_balance_before,
            'receiver_balance_after': receiver_balance_after,
        }

        # 2. Perform Feature Engineering (same as in the notebook)
        input_data['sender_balance_error'] = input_data['sender_balance_after'] + input_data['txn_amount'] - input_data['sender_balance_before']
        input_data['receiver_balance_error'] = input_data['receiver_balance_after'] - input_data['txn_amount'] - input_data['receiver_balance_before']

        # 3. Handle One-Hot Encoding for 'txn_type'
        # The model was trained with these specific columns
        type_cols = ['type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
        for col in type_cols:
            if col == f'type_{txn_type}':
                input_data[col] = 1
            else:
                input_data[col] = 0

        # 4. Create the final DataFrame in the correct column order
        # This order MUST match the order of columns the model was trained on
        column_order = [
            'txn_amount', 'sender_balance_before', 'sender_balance_after',
            'receiver_balance_before', 'receiver_balance_after', 'sender_balance_error',
            'receiver_balance_error', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
        ]
        input_df = pd.DataFrame([input_data])[column_order]

        # 5. Make Prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # --- Display Results ---
        st.subheader("Fraud Analysis Result")
        
        # Display the result with a clear message and icon
        if prediction[0] == 1:
            st.error("High Risk: This transaction is likely FRAUDULENT!", icon="🚨")
        else:
            st.success("Low Risk: This transaction appears to be LEGITIMATE.", icon="✅")
        
        # Display the confidence score
        fraud_probability = prediction_proba[0][1]
        st.metric(label="Fraud Probability Score", value=f"{fraud_probability:.2%}")
        st.progress(fraud_probability)
        
        # Add an expander for more details
        with st.expander("See Detailed Analysis"):
            st.write("Prediction Probabilities:")
            st.write(f"- Legitimate: {prediction_proba[0][0]:.2%}")
            st.write(f"- Fraudulent: {prediction_proba[0][1]:.2%}")
            st.write("---")
            st.write("Data Sent to Model:")
            st.dataframe(input_df)

# --- How It Works Section ---
st.markdown("---")
with st.expander("How does this app work?"):
    st.markdown("""
    This app leverages a machine learning pipeline built with `scikit-learn` and `XGBoost`. Here’s the process:
    1.  **Input Data**: You provide the details of a financial transaction.
    2.  **Feature Engineering**: The app calculates new features that are highly predictive of fraud, such as discrepancies in account balances (`sender_balance_error`, `receiver_balance_error`).
    3.  **Data Transformation**: The data is preprocessed to match the format the model was trained on. This includes scaling numerical values and one-hot encoding the transaction type.
    4.  **Prediction**: The processed data is fed into the pre-trained XGBoost model, which outputs a probability score of the transaction being fraudulent.
    5.  **Result**: The app interprets the score to give a clear, human-readable result.
    """)
