import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load the updated dataset
data = pd.read_csv('updated_advanced_synthetic_cibil_data.csv')

# Drop the ID column for training
X = data.drop(columns=['CIBIL_Score', 'ID'])
y = data['CIBIL_Score']

# Encode categorical features
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Standardize numerical features
scaler = StandardScaler()
X[['Income', 'Debt', 'Credit_History_Length', 'Number_of_Credit_Accounts', 
   'Credit_Utilization', 'Payment_History', 'Default_History', 
   'Age', 'Number_of_Dependents', 'Recent_Credit_Inquiries', 
   'Existing_Savings']] = scaler.fit_transform(X[['Income', 'Debt', 'Credit_History_Length', 
                                                  'Number_of_Credit_Accounts', 
                                                  'Credit_Utilization', 'Payment_History', 
                                                  'Default_History', 'Age', 
                                                  'Number_of_Dependents', 
                                                  'Recent_Credit_Inquiries', 
                                                  'Existing_Savings']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'cibil_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# Load the model and scaler for prediction
model = joblib.load('cibil_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Streamlit interface
st.title("CIBIL Score Predictor")

# User input for prediction
income = st.number_input("Income:", min_value=20000, max_value=150000)
debt = st.number_input("Debt:", min_value=1000, max_value=50000)
credit_history_length = st.number_input("Credit History Length (Years):", min_value=1, max_value=20)
number_of_credit_accounts = st.number_input("Number of Credit Accounts:", min_value=1, max_value=10)
credit_utilization = st.number_input("Credit Utilization Ratio (0 to 1):", min_value=0.0, max_value=1.0)
payment_history = st.number_input("Payment History (%):", min_value=0.0, max_value=100.0)
loan_type = st.selectbox("Loan Type:", ['Personal Loan', 'Home Loan', 'Credit Card', 'Auto Loan'])
default_history = st.number_input("Default History (Number of Defaults):", min_value=0, max_value=5)
employment_status = st.selectbox("Employment Status:", ['Salaried', 'Self-employed', 'Unemployed'])
age = st.number_input("Age:", min_value=18, max_value=65)
residence_status = st.selectbox("Residence Status:", ['Owned', 'Rented', 'Mortgaged'])
education_level = st.selectbox("Education Level:", ['High School', 'Bachelor’s', 'Master’s', 'Doctorate'])
marital_status = st.selectbox("Marital Status:", ['Single', 'Married', 'Divorced'])
number_of_dependents = st.number_input("Number of Dependents:", min_value=0, max_value=5)
recent_credit_inquiries = st.number_input("Recent Credit Inquiries:", min_value=0, max_value=5)
existing_savings = st.number_input("Existing Savings:", min_value=0, max_value=100000)

# Create a DataFrame for input data
input_data = pd.DataFrame({
    'Income': [income],
    'Debt': [debt],
    'Credit_History_Length': [credit_history_length],
    'Number_of_Credit_Accounts': [number_of_credit_accounts],
    'Credit_Utilization': [credit_utilization],
    'Payment_History': [payment_history],
    'Loan_Type': [loan_type],
    'Default_History': [default_history],
    'Employment_Status': [employment_status],
    'Age': [age],
    'Residence_Status': [residence_status],
    'Education_Level': [education_level],
    'Marital_Status': [marital_status],
    'Number_of_Dependents': [number_of_dependents],
    'Recent_Credit_Inquiries': [recent_credit_inquiries],
    'Existing_Savings': [existing_savings]
})

# Encode categorical variables
for column in input_data.select_dtypes(include=['object']).columns:
    input_data[column] = label_encoders[column].transform(input_data[column])

# Scale numerical features
input_data[['Income', 'Debt', 'Credit_History_Length', 'Number_of_Credit_Accounts', 
            'Credit_Utilization', 'Payment_History', 'Default_History', 
            'Age', 'Number_of_Dependents', 'Recent_Credit_Inquiries', 
            'Existing_Savings']] = scaler.transform(input_data[['Income', 'Debt', 'Credit_History_Length', 
                                                                   'Number_of_Credit_Accounts', 
                                                                   'Credit_Utilization', 'Payment_History', 
                                                                   'Default_History', 'Age', 
                                                                   'Number_of_Dependents', 
                                                                   'Recent_Credit_Inquiries', 
                                                                   'Existing_Savings']])

# Make prediction
if st.button("Predict CIBIL Score"):
    prediction = model.predict(input_data)
    st.write(f"Predicted CIBIL Score: {prediction[0]:.2f}")
if st.button("Calculate Accuracy"):
    # Assuming you already have a trained model and X_test, y_test available
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"R-squared (R²): {r2:.2f}")
