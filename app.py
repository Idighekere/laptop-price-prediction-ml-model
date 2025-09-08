# Import libraries for the Streamlit web application, data handling, and model loading
import streamlit as st
import pandas as pd
import joblib

# Loading the pre-trained linear regression model and preprocessing components (encoders and scaler)
model = joblib.load('models_files/linear_regression_model.pkl')
label_encoders = joblib.load('models_files/label_encoders.pkl')
preprocessor = joblib.load('models_files/onehot_preprocessor.pkl')
scaler = joblib.load('models_files/scaler.pkl')

df = pd.read_csv("dataset/laptop_price.csv")

st.title("Laptop Price Predictor")
st.write("Enter your laptop specs to to get a predicted price")


product = st.selectbox("Product", sorted(df["Product"].unique()))  # if used
company = st.selectbox("Company", sorted(df["Company"].unique()))
typename = st.selectbox("Type", sorted(df["TypeName"].unique()))
opsys = st.selectbox("Operating System", sorted(df["OpSys"].unique()))
cpu_brand = st.selectbox("CPU Brand", sorted(df["Cpu Brand"].unique()))
cpu_freq = st.number_input(
    "CPU Frequency (GHz)", min_value=0.5, max_value=5.0, value=2.5
)
gpu_brand = st.selectbox("GPU Brand", sorted(df["Gpu Brand"].unique()))
memory_type = st.selectbox("Memory Type", sorted(df["Memory Type"].unique()))
mem_amount = st.number_input(
    "Memory Amount (GB)", min_value=16, max_value=2048, value=256
)

inches = st.number_input(
    "Screen Size (Inches)", min_value=10.0, max_value=20.0, value=15.6
)
ram = st.slider("RAM (GB)", min_value=2, max_value=64, step=2, value=8)
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0)
screen_width = st.number_input(
    "Screen Width (px)", min_value=800, max_value=4000, value=1920
)
screen_height = st.number_input(
    "Screen Height (px)", min_value=600, max_value=3000, value=1080
)

try:
    if st.button('predict'):
        # Create a DataFrame from user inputs
        input_data = pd.DataFrame({
            'Company': [company],
            'TypeName': [typename],
            'Product': [product],
            'Inches': [inches],
            'Ram': [ram],
            'Weight': [weight],
            'Screen Width': [screen_width],
            'Screen Height': [screen_height],
            'Cpu Brand': [cpu_brand],
            'Cpu Frequency': [cpu_freq],
            'Gpu Brand': [gpu_brand],
            'Memory Amount': [mem_amount],
            'Memory Type': [memory_type],
            'OpSys': [opsys],
        })

        # Encode categorical features using the loaded label encoders
        for col, le in label_encoders.items():
            if col in input_data.columns:
                input_data[col] = le.transform(input_data[col])

        # Apply one-hot encoding to the categorical features
        input_data_encoded = preprocessor.transform(input_data)

        # Scale numerical features using the loaded scaler
        numerical_cols = ['Inches', 'Ram', 'Weight', 'Screen Width', 'Screen Height', 'Cpu Frequency', 'Memory Amount']
        input_data_encoded[numerical_cols] = scaler.transform(input_data_encoded[numerical_cols])

        # Predict the price using the pre-trained model
        predicted_price = model.predict(input_data_encoded)

        st.success(f"The predicted price of the laptop is: ${predicted_price[0]:.2f}")

except Exception as e:

    # Display error message if processing or prediction fails.
    st.error(f"Error: {e}")
    st.write("Please check your input values and try again.")
