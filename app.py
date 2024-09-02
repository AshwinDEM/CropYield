import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import joblib
import os
import pickle
import pandas as pd

numerical_features = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']

with open('./src/model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('./src/label_encoders.pkl', 'rb') as le_file:
    loaded_label_encoders = pickle.load(le_file)

with open('./src/scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# if os.path.exists(model_path):
#     model = joblib.load(model_path)
# else:
#     st.error("Model file not found.")
#     st.stop()

def draw_polygons_around_green(image):
    pixels_per_meter = 10
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([35, 40, 40])    
    upper_green = np.array([85, 255, 255])  
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2) 

    total_green_area_pixels = 0
    for contour in contours:
        area_pixels = cv2.contourArea(contour)  # Calculate the area of the current contour
        total_green_area_pixels += area_pixels  # Add to the total green area
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Draw green polygons
    
    # Convert the total green area from pixels to square meters
    area_per_pixel = (1 / pixels_per_meter) ** 2  # Convert pixels to square meters
    total_green_area_m2 = total_green_area_pixels * area_per_pixel
    
    # Convert the area from square meters to acres
    total_green_area_acres = total_green_area_m2 / 4046.86
    
    return image, total_green_area_acres

def main():
    st.title("Crop Yield Prediction")
    st.write("Upload an image of crops, and the model will detect the crops and predict the yield.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        image = cv2.imread(temp_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        st.image(image, caption="Uploaded Image", use_column_width=True)

        processed_image, area_acres = draw_polygons_around_green(image.copy())

        st.image(processed_image, caption="Processed Image", use_column_width=True)
        st.write(f"Total crop area in acres: {area_acres}")

        # prediction = loaded_model.predict(processed_image)

        # User input for new data
        # drop down for area and item, input for year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp
        area = st.selectbox("Area", ['India', 'Japan', 'Albania', 'Spain'])
        item = st.selectbox("Item", ['Cassava', 'Maize', 'Rice, paddy', 'Wheat'])
        year = st.number_input("Year", min_value=1961, max_value=2100)
        avg_rain_fall = st.number_input("Average Rain Fall (mm/year)", min_value=0.0, max_value=10000.0)
        pesticides = st.number_input("Pesticides (tonnes)", value = 10)
        avg_temp = st.number_input("Average Temperature (°C)", value = 20)

        new_data = {
            'Area': [area],
            'Item': [item],
            'Year': [year],
            'average_rain_fall_mm_per_year': [avg_rain_fall],
            'pesticides_tonnes': [pesticides],
            'avg_temp': [avg_temp]
        }

        # Convert new data to DataFrame
        input_df = pd.DataFrame(new_data)

        # Encode categorical features using loaded label encoders
        for col, le in loaded_label_encoders.items():
            input_df[col] = le.transform(input_df[col])

        # Scale numerical features using loaded scaler
        input_df[numerical_features] = loaded_scaler.transform(input_df[numerical_features])

        # Make prediction with the loaded model
        prediction = loaded_model.predict(input_df)

        # Print the predicted yield
        print(f"Predicted yield: {prediction[0]}")

        area_hectares = area_acres * 0.404686
        
        st.write("The predicted yield is = ")
        st.write(prediction[0] * area_hectares, "hg")
    else:
        st.write("Please upload an image to continue.")

if __name__ == "__main__":
    main()
