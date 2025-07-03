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
# This is the final list of features AFTER one-hot encoding
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
            # 1. Start with a dictionary of the raw inputs
            input_data = {
                'txn_amount': txn_amount,
                'sender_balance_before': sender_balance_before,
                'sender_balance_after': sender_balance_after,
                'receiver_balance_before': receiver_balance_before,
                'receiver_balance_after': receiver_balance_after,
            }

            # 2. Engineer the features using the CORRECT names from your notebook
            input_data['bal_diff_sender'] = input_data['sender_balance_after'] + input_data['txn_amount'] - input_data['sender_balance_before']
            input_data['bal_diff_receiver'] = input_data['receiver_balance_after'] - input_data['txn_amount'] - input_data['receiver_balance_before']

            # 3. Manually perform one-hot encoding, just like pd.get_dummies
            # Note: 'CASH_IN' was the first category, so it was dropped and is represented by all 0s.
            type_cols = ['type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
            for col in type_cols:
                if col == f'type_{txn_type}':
                    input_data[col] = 1
                else:
                    input_data[col] = 0

            # 4. Create the final DataFrame
            input_df = pd.DataFrame([input_data])

            # 5. Enforce the exact column order
            input_df = input_df[MODEL_COLUMN_ORDER]

            # 6. Make Prediction with the fully preprocessed data
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
                st.write("Final Data Sent to Model (Fully Preprocessed):")
                st.dataframe(input_df)

# --- How It Works Section ---
st.markdown("---")
with st.expander("How does this app work?"):
    st.markdown("""
    1.  **Input Data**: You provide the raw transaction details.
    2.  **Manual Preprocessing**: The app meticulously replicates the preprocessing from the training notebook:
        - It creates the `bal_diff_...` features.
        - It performs one-hot encoding manually to create the `type_...` columns.
        - It enforces the exact column order the model was trained on.
    3.  **Prediction**: This fully prepared data is then fed to the saved pipeline, which handles the final scaling step and makes the prediction.
    """)
