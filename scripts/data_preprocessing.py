import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()
    temp=data[['Savingaccounts','Checkingaccount']]
    data.columns = data.columns.str.strip()
    # Handle missing values
    # data = data.dropna()
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    data['Risk'] = label_encoder.fit_transform(data['Risk'])
    
    # Prepare features and target
    X = data.drop('Risk', axis=1)
    y = data['Risk']
    
    # Handle categorical features
    X = pd.get_dummies(X, drop_first=True)
   
    X=pd.concat([X,temp],axis=1)
    return X, y
