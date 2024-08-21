# Credit Risk Modeling

This project involves building a credit risk model using various machine learning techniques.

## File Structure

- `app.py`: Flask application for serving the model.
- `data/credit_risk_data.csv`: Dataset.
- `model/credit_risk_model.pkl`: Saved model.
- `model/scaler.pkl`: Saved scaler.
- `scripts/`: Contains scripts for data preprocessing, feature engineering, model training, evaluation, hyperparameter tuning, and interpretation.
- `tests/`: Contains test scripts for validating the code.
- `requirements.txt`: Python package dependencies.
- `Procfile`: Heroku deployment configuration.
- `README.md`: Project documentation.

## Installation

1. Clone the repository.
2. Install the requirements:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the training script:

    ```bash
    python scripts/model_training.py
    ```

4. Start the Flask application:

    ```bash
    python app.py
    ```

## Usage

Send a POST request to `/predict` with the following JSON payload:

```json
{
    "Age": 30,
    "Sex": "male",
    "Job": "skilled",
    "Housing": "own",
    "Saving accounts": "little",
    "Checking account": "moderate",
    "Credit amount": 5000,
    "Duration": 24,
    "Purpose": "car",
    "Saving_Checking_Account": "little_moderate"
}
