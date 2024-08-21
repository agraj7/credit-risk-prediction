import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from scripts.data_preprocessing import preprocess_data
from scripts.feature_engineering import feature_engineering
from sklearn.model_selection import train_test_split

def test_model():
    # Load data
    X, y = preprocess_data('data/credit_risk_data.csv')
    X = feature_engineering(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load model
    model = joblib.load('model/credit_risk_model.pkl')
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.7, "Model accuracy is below the acceptable threshold"

if __name__ == "__main__":
    test_model()
