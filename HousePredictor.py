import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset (replace 'file.csv' with the actual file path)
data = pd.read_csv('bangalore_house_prices.csv')

# Example columns: ['Locality', 'Area', 'Rooms', 'Bathrooms', 'Price']
# Feature and target separation
X = data[['Locality', 'Area', 'Rooms', 'Bathrooms']]
y = data['Price']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining categorical and numerical features
categorical_features = ['Locality']
numerical_features = ['Area', 'Rooms', 'Bathrooms']

# Preprocessing pipelines
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine preprocessors in a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define the model
model = LinearRegression()

# Create the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Train the model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# Example: Predict a house price
example_input = pd.DataFrame({
    'Locality': ['Whitefield'],
    'Area': [1200],
    'Rooms': [3],
    'Bathrooms': [2]
})

predicted_price = pipeline.predict(example_input)
print(f"Predicted Price: {predicted_price[0]}")
