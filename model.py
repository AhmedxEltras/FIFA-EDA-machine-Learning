import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import joblib

# Load the FIFA 19 player data
df = pd.read_csv("fifa_eda.csv")

# Data Cleaning: Drop rows with missing values in the selected columns
df = df.dropna(subset=['Value', 'Overall', 'Potential', 'Age', 'Wage', 'Height', 'Weight'])

# Convert 'Value' and 'Wage' from string to numeric
df['Value'] = df['Value'].replace('[\€,]', '', regex=True).astype(float)
df['Wage'] = df['Wage'].replace('[\€,]', '', regex=True).astype(float)

# Select features and target variable
features = ['Overall', 'Potential', 'Age', 'Wage', 'Height', 'Weight']
target = 'Value'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the KNN model
knn_model = KNeighborsRegressor(n_neighbors=5)

# Train the model
knn_model.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully!")
