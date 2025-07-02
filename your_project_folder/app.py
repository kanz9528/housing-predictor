import streamlit as st
import os
import pandas as pd

st.write("📂 Current directory:", os.getcwd())
st.write("📄 Files:", os.listdir())

if not os.path.exists("your_project_folder/Housing.csv"):
    st.error("❌ File 'Housing.csv' not found.")
    st.stop()

df = pd.read_csv("Housing.csv")
st.success("✅ File loaded successfully!")
import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os


# ----------------------
# Train & Save Model (run once if model doesn't exist)
# ----------------------
MODEL_PATH = "housing_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.info("Training model for the first time...")

    # Load dataset
    df = pd.read_csv(r"Housing.csv")  # Make sure Housing.csv is in the same folder

    # Select features and target
    X = df[['area', 'bedrooms', 'bathrooms', 'stories']]
    y = df['price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, MODEL_PATH)
    st.success("✅ Model trained and saved as housing_model.pkl")
else:
    model = joblib.load(MODEL_PATH)

# ----------------------
# Streamlit Web App
# ----------------------
st.title("🏠 Housing Price Prediction App")
st.sidebar.header("Enter House Details")

# Sidebar input fields
area = st.sidebar.number_input("Area (sq ft)", min_value=500, max_value=10000, value=2500)
bedrooms = st.sidebar.slider("Bedrooms", 1, 5, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 4, 2)
stories = st.sidebar.slider("Stories", 1, 4, 2)

# Prediction
if st.button("Predict Price"):
    input_df = pd.DataFrame([[area, bedrooms, bathrooms, stories]],
                            columns=['area', 'bedrooms', 'bathrooms', 'stories'])
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted House Price: ₹{int(prediction):,}")
