import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from data_preprocessing import preprocess_data
from feature_engineering import feature_engineering
from sklearn.model_selection import train_test_split

def evaluate_model(file_path):
    # Load data
    X, y = preprocess_data(file_path)
    X = feature_engineering(X)
    
    # Load model and scaler
    model = joblib.load('model/credit_risk_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Print evaluation metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob)}")

if __name__ == "__main__":
    evaluate_model('data/credit_risk_data.csv')
