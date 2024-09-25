import streamlit as st
import requests
import pandas as pd

# Set the title of the app
st.title("Churn Prediction")

# Create a file uploader for the CSV file
uploaded_file = st.file_uploader("Upload your CSV file:", type="csv")

# Define the FastAPI prediction endpoint
API_URL = "http://127.0.0.1:8000/predict/"  # Adjust this URL as necessary

# Create a button that when clicked, will send the file to the API
if st.button("Submit"):
    if uploaded_file is not None:
        try:
            # Make a POST request to the FastAPI API with the uploaded file
            response = requests.post(API_URL, files={"file": (uploaded_file.name, uploaded_file.getvalue())})

            # Check if the request was successful
            if response.status_code == 200:
                # Parse and display the results
                result = response.json()
                st.success("Predictions received!")
                
                # Display predictions for each model
                st.write("### Logistic Regression Predictions")
                st.write(result['Logistic Regression Predictions'])

                st.write("### Random Forest Predictions")
                st.write(result['Random Forest Predictions'])

                st.write("### XGBoost Predictions")
                st.write(result['XGBoost Predictions'])

            else:
                st.error("Error in prediction request: " + str(response.status_code) + " - " + response.text)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload a CSV file before clicking submit.")
