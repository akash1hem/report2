import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample data creation
data = {
    'Square_Feet': [1500, 1800, 2400, 3000, 3500, 4000, 4500, 5000],
    'Bedrooms': [3, 4, 3, 5, 4, 6, 5, 6],
    'Age': [10, 15, 20, 5, 2, 8, 3, 1],
    'Price': [300000, 350000, 450000, 500000, 600000, 650000, 700000, 800000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Define features and target
X = df[['Square_Feet', 'Bedrooms', 'Age']]
y = df['Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print (f'Mean Squared Error: {mse}')
print (f'R^2 Score: {r2}')

# Plotting actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2) # Line of equality
plt.show()
