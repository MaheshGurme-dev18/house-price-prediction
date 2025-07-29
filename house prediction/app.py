import streamlit as st
import pickle
import pandas as pd

# Load the trained model and column names
model = pickle.load(open("house_prediction_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))  # List of all input features

# Extract location columns (everything except these)
location_cols = [col for col in columns if col not in ['total_sqft', 'bath', 'bhk']]

# Streamlit App UI
st.title("🏠 Bengaluru House Price Prediction")
st.markdown("Provide property details to predict the price (in lakhs).")

# Inputs
location = st.selectbox("Select Location", sorted(location_cols))
sqft = st.number_input("Total Square Feet", min_value=300)
bath = st.selectbox("Number of Bathrooms", [1, 2, 3, 4, 5])
bhk = st.selectbox("Number of Bedrooms (BHK)", [1, 2, 3, 4, 5])

# Predict Button
if st.button("Predict Price"):
    # Build input data
    input_data = {col: 0 for col in columns}
    input_data['total_sqft'] = float(sqft)
    input_data['bath'] = float(bath)
    input_data['bhk'] = float(bhk)
    if location in input_data:
        input_data[location] = 1.0

    # Convert to DataFrame and ensure numeric
    input_df = pd.DataFrame([input_data]).astype(float)

    # Make prediction
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: ₹ {round(prediction, 2)} Lakhs")











    



   


