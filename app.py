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
MODEL_COLUMN_ORDER = [
    'txn_amount', 'sender_balance_before', 'sender_balance_after',
    'receiver_balance_before', 'receiver_balance_after',
    'bal_diff_sender', 'bal_diff_receiver',
    'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
]

# --- Caching the Model ---
@st.cache_resource
def load_model():
    """Load the pre-trained pipeline (Scaler + XGBoost)."""
    try:
        model = joblib.load('fraud_detection_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'fraud_detection_model.pkl' is in the root directory.")
        return None

# Load the model
model = load_model()

# --- Initialize Session State for Inputs ---
# This allows us to programmatically change the widget values
if 'input_data' not in st.session_state:
    st.session_state.input_data = {
        'txn_type': 'PAYMENT', 'txn_amount': 1000.0,
        'sender_balance_before': 50000.0, 'sender_balance_after': 49000.0,
        'receiver_balance_before': 10000.0, 'receiver_balance_after': 11000.0
    }

# --- Functions to update session state ---
def load_fraud_example():
    st.session_state.input_data = {
        'txn_type': 'CASH_OUT', 'txn_amount': 250000.0,
        'sender_balance_before': 250000.0, 'sender_balance_after': 0.0,
        'receiver_balance_before': 50000.0, 'receiver_balance_after': 50000.0
    }

# --- Application Header ---
st.title("🛡️ Real-Time Transaction Fraud Detection")
st.markdown("""
This application uses a sophisticated XGBoost model to predict fraudulent transactions.
Enter transaction details manually or load an example to see the model in action.
""")

# --- Main Page Inputs in Columns ---
st.divider()
st.subheader("Enter Transaction Details to Analyze 💳")

# --- Example Button ---
if st.button("Load a Known Fraudulent Example", on_click=load_fraud_example):
    pass # The on_click callback handles the logic

col1, col2 = st.columns(2)
with col1:
    # Use st.session_state values to control the widgets
    txn_type = st.selectbox('Transaction Type', ('PAYMENT', 'CASH_OUT', 'TRANSFER', 'CASH_IN', 'DEBIT'), 
                            index=['PAYMENT', 'CASH_OUT', 'TRANSFER', 'CASH_IN', 'DEBIT'].index(st.session_state.input_data['txn_type']))
    sender_balance_before = st.number_input('Sender Balance Before', min_value=0.0, format="%.2f", value=st.session_state.input_data['sender_balance_before'])
    receiver_balance_before = st.number_input('Receiver Balance Before', min_value=0.0, format="%.2f", value=st.session_state.input_data['receiver_balance_before'])

with col2:
    txn_amount = st.number_input('Transaction Amount', min_value=0.0, format="%.2f", value=st.session_state.input_data['txn_amount'])
    sender_balance_after = st.number_input('Sender Balance After', min_value=0.0, format="%.2f", value=st.session_state.input_data['sender_balance_after'])
    receiver_balance_after = st.number_input('Receiver Balance After', min_value=0.0, format="%.2f", value=st.session_state.input_data['receiver_balance_after'])

st.divider()

# --- Prediction Logic ---
if st.button("Analyze Transaction", type="primary"):
    if model is not None:
        input_data = {
            'txn_amount': txn_amount, 'sender_balance_before': sender_balance_before,
            'sender_balance_after': sender_balance_after, 'receiver_balance_before': receiver_balance_before,
            'receiver_balance_after': receiver_balance_after,
        }

        input_data['bal_diff_sender'] = input_data['sender_balance_after'] + input_data['txn_amount'] - input_data['sender_balance_before']
        input_data['bal_diff_receiver'] = input_data['receiver_balance_after'] - input_data['txn_amount'] - input_data['receiver_balance_before']

        type_cols = ['type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
        for col in type_cols:
            if col == f'type_{txn_type}':
                input_data[col] = 1
            else:
                input_data[col] = 0

        input_df = pd.DataFrame([input_data])
        input_df = input_df[MODEL_COLUMN_ORDER]

        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Display Results
        st.subheader("Fraud Analysis Result")
        if prediction[0] == 1:
            st.error("High Risk: This transaction is likely FRAUDULENT!", icon="🚨")
        else:
            st.success("Low Risk: This transaction appears to be LEGITIMATE.", icon="✅")

        fraud_probability = prediction_proba[0][1]
        st.metric(label="Fraud Probability Score", value=f"{fraud_probability:.2%}")
        st.progress(float(fraud_probability))
        
        st.divider()
        st.subheader("Detailed Analysis")
        st.write("**Prediction Probabilities:**")
        st.write(f"- Legitimate: `{prediction_proba[0][0]:.2%}`")
        st.write(f"- Fraudulent: `{prediction_proba[0][1]:.2%}`")
        st.write("**Final Data Sent to Model (Fully Preprocessed):**")
        st.dataframe(input_df, use_container_width=True)
        
# --- How It Works Section ---
st.markdown("---")
# This can also be removed if you feel the app is self-explanatory now
with st.expander("How does this app work?"):
    st.markdown("""
    1.  **Input Data**: You provide the raw transaction details.
    2.  **Manual Preprocessing**: The app meticulously replicates the preprocessing from the training notebook.
    3.  **Prediction**: This fully prepared data is then fed to the saved pipeline for prediction.
    """)
