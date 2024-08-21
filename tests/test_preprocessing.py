import pandas as pd
from scripts.data_preprocessing import preprocess_data

def test_preprocessing():
    X, y = preprocess_data('data/credit_risk_data.csv')
    
    # Check if there are missing values
    assert X.isnull().sum().sum() == 0, "Missing values found in the dataset"
    
    # Check if the target variable is correctly encoded
    assert set(y.unique()).issubset({0, 1}), "Target variable 'Risk' is not correctly encoded"

if __name__ == "__main__":
    test_preprocessing()
