import shap
import joblib
import pandas as pd
from data_preprocessing import preprocess_data
from feature_engineering import feature_engineering
def interpret_model(file_path):
    # Load the model and scaler
    model = joblib.load('model/credit_risk_model.pkl')
    scaler = joblib.load('model/scaler.pkl')

    # Load and preprocess data
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()  # Strip any extra spaces
    X, _ = preprocess_data(file_path)
    
    # Apply feature engineering
    X = feature_engineering(X)
    
    # Convert all columns to numeric types
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Drop rows with NaN values after conversion
    X = X.dropna()
    
    # Ensure that model and data types are compatible
    X = scaler.transform(X)
    
    # Use SHAP for model interpretation
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    
    # Save or visualize SHAP values as needed
    shap.summary_plot(shap_values, X)

if __name__ == "__main__":
    interpret_model('data/credit_risk_data.csv')
