from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from data_preprocessing import preprocess_data
from feature_engineering import feature_engineering
from sklearn.model_selection import train_test_split

def rf_evaluate(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42
    )
    model.fit(X_train, y_train)
    return cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=5).mean()

def tune_hyperparameters(file_path):
    # Load and preprocess data
    X, y = preprocess_data(file_path)
    X = feature_engineering(X)
    
    # Split data
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define hyperparameter space
    param_bounds = {
        'n_estimators': (50, 200),
        'max_depth': (5, 50),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 20)
    }
    
    # Initialize Bayesian Optimization
    optimizer = BayesianOptimization(f=rf_evaluate, pbounds=param_bounds, random_state=42)
    optimizer.maximize(init_points=5, n_iter=20)
    
    print("Best Parameters:", optimizer.max)

if __name__ == "__main__":
    tune_hyperparameters('data/credit_risk_data.csv')
