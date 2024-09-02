# import streamlit as st

# import cv2
# import numpy as np
# from PIL import Image
# import tempfile


# def draw_polygons_around_green(image):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
#     lower_green = np.array([35, 40, 40])    
#     upper_green = np.array([85, 255, 255])  
    
#     mask = cv2.inRange(hsv, lower_green, upper_green)
    
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     cv2.drawContours(image, contours, -1, (0, 255, 0), 2) 
    
#     return image

# def main():
#     st.title("Crop Yield Prediction")
#     st.write("Upload an image of crops, and the model will detect the crops and predict the yield.")

#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#             temp_file.write(uploaded_file.read())
#             temp_path = temp_file.name

#         image = cv2.imread(temp_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

#         st.image(image, caption="Uploaded Image", use_column_width=True)

#         processed_image = draw_polygons_around_green(image.copy())

#         st.image(processed_image, caption="Processed Image", use_column_width=True)
        
#         temp = 0

#         st.write("The predicted yield is = ")
#         st.write(temp)
#     else :
#         st.write("Please upload an image to continue.")

# if __name__ == "__main__":
#     main()

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import joblib
import os

# Define the path to the model
model_path = 'models\model.pkl'

# Check if the model file exists
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("Model file not found.")
    st.stop()

def draw_polygons_around_green(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([35, 40, 40])    
    upper_green = np.array([85, 255, 255])  
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2) 
    
    return image

# def preprocess_image(image):
#     # Example preprocessing: resizing and converting to a 1D array
#     image = cv2.resize(image, (224, 224))  # Adjust size as needed
#     image = image / 255.0  # Normalize if required
#     image = image.flatten()  # Convert to 1D array for prediction
#     return np.array([image])

def preprocess_image(image):
    """
    Preprocess the image to match the expected number of features (e.g., 21).
    """
    # Resize the image to a fixed size (e.g., 64x64), if needed
    image = cv2.resize(image, (64, 64))  # Adjust size if necessary
    
    # Convert to grayscale (if your model expects grayscale images)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Flatten the image and normalize
    image = image.flatten() / 255.0
    
    # Check if we need to extract or adjust to 21 features
    if len(image) == 21:  # Adjust this check based on actual feature size needed
        return np.array([image])
    else:
        # Example of extracting a fixed number of features (if needed)
        # You may need to perform additional feature extraction or selection
        # For example, compute statistics or extract specific features
        features = image[:21]  # Example: use the first 21 features
        return np.array([features])


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

        processed_image = draw_polygons_around_green(image.copy())

        st.image(processed_image, caption="Processed Image", use_column_width=True)
        
        # Preprocess the image for model prediction
        preprocessed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(preprocessed_image)
        
        st.write("The predicted yield is = ")
        st.write(prediction[0])
    else:
        st.write("Please upload an image to continue.")

if __name__ == "__main__":
    main()
