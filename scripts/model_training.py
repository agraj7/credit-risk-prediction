import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_preprocessing import preprocess_data
from feature_engineering import feature_engineering

def train_model(file_path):
    # Load and preprocess data
    X, y = preprocess_data(file_path)
    
    # Feature engineering
    X = feature_engineering(X)
    X.to_csv("riskData.csv",index=False)
    print(X.columns)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize and train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'model/credit_risk_model.pkl')
    
    # Train and save scaler
    scaler = StandardScaler().fit(X_train)
    joblib.dump(scaler, 'model/scaler.pkl')

if __name__ == "__main__":
    train_model('data/credit_risk_data.csv')
