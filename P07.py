import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
data = pd.read_csv('iris.csv')
print(data.head())
print(data.columns)
# Display basic info about the dataset
print("\nDataset Info:")
print(data.info())

# Display summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Encode the 'Species' column as numeric values (0, 1, 2)
label_encoder = LabelEncoder()
data['variety'] = label_encoder.fit_transform(data['variety'])


# Define the features (X) and target variable (Y)
X = data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]

Y = data['variety']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)

# Feature scaling for better model performance (especially for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)

# Make predictions
prediction = model.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(Y_test, prediction)
print("\nAccuracy of Logistic Regression model:", accuracy)

# Optionally, display the predictions
print("\nPredictions:", prediction)
