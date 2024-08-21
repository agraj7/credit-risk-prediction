import pandas as pd

def feature_engineering(df):
    # Strip spaces from column names
    df.columns = df.columns.str.strip()
    
    # Check if required columns exist
    required_columns = ['Age', 'Creditamount', 'Duration']
    for column in required_columns:
        if column not in df.columns:
            raise KeyError(f"Required column '{column}' is missing from the DataFrame")

    # Example feature engineering
    df['Age_Income_Ratio'] = df['Age'] / (df['Creditamount'] + 1)  # Adding 1 to avoid division by zero
    df['Credit_Duration_Ratio'] = df['Creditamount'] / (df['Duration'] + 1)
    
    # Example of combining 'Saving accounts' and 'Checking account' into a single feature
    # if 'Savingaccounts' in df.columns and 'Checkingaccount' in df.columns:
    df['Saving_Checking_Account'] = df['Savingaccounts'].astype(str) + '_' + df['Checkingaccount'].astype(str)
    
    # One-hot encode new categorical features
    df = pd.get_dummies(df, columns=['Saving_Checking_Account'], drop_first=True)
    
    df=df.drop(['Savingaccounts','Checkingaccount'],axis=1)
    return df
