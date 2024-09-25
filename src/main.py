# Importing required libraries
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Create FastAPI app
app = FastAPI()

# Define the prediction function
def train_and_predict(data):
    # Data Preprocessing
    # Replace any missing values with the median
    data.fillna(data.median(), inplace=True)
    
    # Splitting features and target variable
    X = data.drop(['churn'], axis=1)
    y = data['churn']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardizing the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initializing models
    logreg = LogisticRegression(random_state=42)
    rf = RandomForestClassifier(random_state=42)
    xgb = XGBClassifier(random_state=42)

    # Training models
    logreg.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    # Making predictions
    logreg_preds = logreg.predict(X_test)
    rf_preds = rf.predict(X_test)
    xgb_preds = xgb.predict(X_test)

    return {
        "Logistic Regression Predictions": logreg_preds.tolist(),
        "Random Forest Predictions": rf_preds.tolist(),
        "XGBoost Predictions": xgb_preds.tolist()
    }

# Define an endpoint for file upload and prediction
@app.post("/predict/")
async def predict_churn(file: UploadFile = File(...)):
    # Read the uploaded CSV file into a DataFrame
    try:
        df = pd.read_csv(file.file)

        # Check if the required target column is present
        if 'churn' not in df.columns:
            return JSONResponse(content={"error": "The 'churn' column is required in the data."}, status_code=400)

        # Train models and make predictions
        predictions = train_and_predict(df)

        return predictions

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# To run the FastAPI app, uncomment the following lines
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
