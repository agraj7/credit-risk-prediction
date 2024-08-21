import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model = joblib.load('model/credit_risk_model.pkl')
scaler = joblib.load('model/scaler.pkl')

def preprocess_input(data):
    """
    Preprocess and scale the input data.
    This function should match the preprocessing and feature engineering steps
    used during model training.
    """
    df = pd.DataFrame([data])
    
    # Feature engineering steps
    df['Age_Income_Ratio'] = df['Age'] / (df['Creditamount'] + 1)  # Avoid division by zero
    df['Credit_Duration_Ratio'] = df['Creditamount'] / (df['Duration'] + 1)
    df['Saving_Checking_Account'] = df['Savingaccounts'] + '_' + df['Checkingaccount']
    
    # Encode categorical features
    df = pd.get_dummies(df, drop_first=True)
    
    # Align the columns with the model's expected input
    expected_columns = pd.read_csv('riskData.csv')
    expected_columns = pd.get_dummies(expected_columns, drop_first=True).columns
    df = df.reindex(columns=expected_columns, fill_value=0)

    # Scale the data
    X = scaler.transform(df)
    
    return X

# Streamlit app
st.title("Credit Risk Prediction")

# User input form
with st.form("input_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    job = st.selectbox("Job", ["unskilled", "skilled", "highly skilled"])
    housing = st.selectbox("Housing", ["own", "rent", "free"])
    saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"])
    checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"])
    credit_amount = st.number_input("Credit Amount", min_value=0, value=5000)
    duration = st.number_input("Duration (months)", min_value=1, value=24)
    purpose = st.selectbox("Purpose", ["car", "furniture", "radio/tv", "domestic appliance", "repairs", "education", "vacation", "retraining", "business"])

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare input data
        input_data = {
            "Age": age,
            "Sex": sex,
            "Job": job,
            "Housing": housing,
            "Savingaccounts": saving_accounts,
            "Checkingaccount": checking_account,
            "Creditamount": credit_amount,
            "Duration": duration,
            "Purpose": purpose
        }
        
        # Preprocess and make prediction
        X = preprocess_input(input_data)
        prediction = model.predict(X)
        
        # Display the result
        risk = "High Risk" if prediction[0] == 1 else "Low Risk"
        st.write(f"The predicted credit risk is: **{risk}**")
