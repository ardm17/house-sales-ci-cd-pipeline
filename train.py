import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import skops.io as sio
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("Data/kc_house_data.csv")

# Selected features
X = df[['sqft_living', 'bedrooms', 'bathrooms', 'floors', 'condition', 'grade', 'waterfront', 'yr_built']]
y = df['price']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f"Mean Absolute Error: {mae:.2f}, Mean Squared Error: {mse:.2f}")

# Save results
if not os.path.exists("Results"):
    os.makedirs("Results")

with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"Mean Absolute Error: {mae:.2f}, Mean Squared Error: {mse:.2f}")

# Save the model
if not os.path.exists("Model"):
        os.makedirs("Model")

sio.dump(model, "Model/house_rf_model.skops")

print("Model saved as 'house_rf_model.skops'.")
