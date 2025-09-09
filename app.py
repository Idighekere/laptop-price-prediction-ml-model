import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and preprocessors
model = joblib.load('models_files/linear_regression_model.pkl')
preprocessor = joblib.load('models_files/onehot_preprocessor.pkl')
label_encoders = joblib.load('models_files/label_encoders.pkl')
scaler = joblib.load('models_files/scaler.pkl')

# Reference DataFrame for value options
df = pd.read_csv("dataset/laptop_price.csv", encoding='latin-1')


# UI: Gather user input

st.markdown("""
# 💻 Laptop Price Predictor

Enter your laptop’s specs below to get an instant price estimate.
Just fill in the details and click **Predict Price** to see your laptop’s expected value.

""")

st.subheader("Laptop Identification")
company = st.selectbox("Company", sorted(df["Company"].unique()))
# Filter products for selected company
filtered_products = df[df["Company"] == company]["Product"].unique()
product = st.selectbox("Product", sorted(filtered_products))
typename = st.selectbox("Type", sorted(df["TypeName"].unique()))
opsys = st.selectbox("Operating System", sorted(df["OpSys"].unique()))
st.divider()

st.subheader("Display Specifications")
inches = st.number_input("Screen Size (Inches)", min_value=10.0, max_value=20.0, value=15.6)
screen_resolution = st.selectbox("Screen Resolution (px)", sorted(df["ScreenResolution"].str.split(' ').apply(lambda x:x[-1]).unique()))
screen_width = int(screen_resolution.split('x')[0])
screen_height = int(screen_resolution.split('x')[1])
touchscreen = st.checkbox("Is the screen a touchscreen?")
touchscreen_value = int(touchscreen)
st.divider()

st.subheader("Hardware Specifications")
ram = st.selectbox("RAM (GB)", sorted(df["Ram"].unique()))
ram = int(ram.replace('GB',''))
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0)
st.divider()

st.subheader("Processor and Graphics")
cpu_brand = st.selectbox("CPU Brand", sorted(df["Cpu"].str.split(" ").str[0].unique()))
cpu_freq = st.number_input("CPU Frequency (GHz)", min_value=0.5, max_value=5.0, value=2.5)
gpu_brand = st.selectbox("GPU Brand", sorted(df["Gpu"].str.split(" ").str[0].unique()))
st.divider()

st.subheader("Storage")
memory_type = st.selectbox("Storage Type", sorted(df["Memory"].str.split(" ").str[1].unique()))

df["Memory Amount"] = df["Memory"].str.split(" ").apply(lambda x: x[0])

def make_all_memory_in_GB(value):
    if 'TB' in value:
        return float(value[:value.find('TB')]) * 1000
    elif  'GB' in value:
        return float(value[:value.find('GB')])


mem_amount_input = st.selectbox("Storage Amount", sorted(df["Memory Amount"].unique()))
mem_amount = make_all_memory_in_GB(mem_amount_input)
st.divider()

# List of columns (must match training)
feature_cols = [
    'Company', 'TypeName', 'OpSys', 'Product',
    'Inches', 'Ram', 'Weight', 'Touch Screen',
    'Screen Width', 'Screen Height',
    'Cpu Brand', 'Cpu Frequency', 'Gpu Brand',
    'Memory Amount', 'Memory Type'
]
numerical_cols = [
    'Inches', 'Ram', 'Weight', 'Screen Width', 'Screen Height', 'Touch Screen',
    'Cpu Frequency', 'Memory Amount',
]

# Prepare DataFrame
input_data = pd.DataFrame([[
    company, typename, opsys, product,
    inches, ram, weight, touchscreen_value,
    screen_width, screen_height,
    cpu_brand, cpu_freq, gpu_brand,
    mem_amount, memory_type
]], columns=feature_cols)

if st.button('Predict Price'):
    # print(scaler_features)
    try:

        # Reorder columns to match scaler
        input_numeric = input_data[numerical_cols]

        # Now scale
        scaled_numeric = scaler.transform(input_numeric)

        # Put scaled values back
        input_data[numerical_cols] = scaled_numeric


        # Encode Binary column
        # print('Label Encoders', label_encoders.keys())
        for col, le in label_encoders.items():
            if col in input_data.columns:
                input_data[col] = le.transform(input_data[col])


        # Use the saved preprocessor to encode nominal categorical columns
        # x_input = input_data.drop("Price_euros", axis=1, errors="ignore")
        X_processed = preprocessor.transform(input_data)
        # print("X Processed shape", X_processed.shape)
        # print("X Processed shape", X_processed)

        with st.spinner("Predicting laptop price..."):
            pred = model.predict(X_processed)
        st.success(f"Predicted price: €{pred[0]:,.2f}")
        st.toast("Success!", icon="✅")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.toast("An error occurred!", icon="❌")
