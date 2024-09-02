import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Sample data
data = {
    'Area': ['Africa', 'Asia', 'Europe', 'Africa'],
    'Item': ['Cassava', 'Wheat', 'Barley', 'Cassava'],
    'Year': [2000, 2001, 2002, 2003],
    'average_rain_fall_mm_per_year': [1000, 1100, 1050, 1200],
    'pesticides_tonnes': [100, 150, 120, 130],
    'avg_temp': [30, 25, 20, 28],
    'Yield': [10, 20, 15, 18]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Separate features and target
X = df.drop('Yield', axis=1)
y = df['Yield']

# Initialize LabelEncoders for categorical features
label_encoders = {
    'Area': LabelEncoder(),
    'Item': LabelEncoder()
}

# Encode categorical features
for col, le in label_encoders.items():
    X[col] = le.fit_transform(X[col])

# Initialize and fit Scaler
scaler = StandardScaler()
numerical_features = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Train the RandomForest model
model = RandomForestRegressor()
model.fit(X, y)

# Save the model, label encoders, and scaler
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump(label_encoders, le_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Load the model, label encoders, and scaler
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('label_encoders.pkl', 'rb') as le_file:
    loaded_label_encoders = pickle.load(le_file)

with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# New data for prediction
new_data = {
    'Area': ['Africa'],
    'Item': ['Cassava'],
    'Year': [2000],
    'average_rain_fall_mm_per_year': [1000],
    'pesticides_tonnes': [100],
    'avg_temp': [30]
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
