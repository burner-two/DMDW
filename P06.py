import numpy as np
import pandas as pd
from scipy import stats

# Load the dataset
data = pd.read_csv('data.csv')
print("Dataset:\n", data)

# Assign 'tcs' as the independent variable (X)
x = data['tcs']

# Assign 'wipro' as the dependent variable (Y)
y = data['wipro']

# Perform linear regression using scipy's linregress
n = stats.linregress(x, y)
print("\nLinear Regression Output:\n", n)

# Extract slope (m) from the regression result
s = n.slope
print("\nSlope (m):", s)

# Extract intercept (c) from the regression result
i = n.intercept
print("Intercept (c):", i)

# Use the regression line equation: y = mx + c to make predictions
pred_y = s * x + i
print("\nPredicted 'wipro' values:\n", pred_y)

# Calculate R-squared value (coefficient of correlation)
r_squared = n.rvalue
print("\nR-squared (correlation coefficient):", r_squared)
