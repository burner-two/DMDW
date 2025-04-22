import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv('spam_ham_dataset.csv', encoding="ISO-8859-1")
print(data.head())

# Check label distribution
print("Label counts:\n", data["label"].value_counts())

# Clean text column (optional trimming, here it's minimal)
text = []
for i in range(len(data)):
    ln = data["text"][i]
    line = ""
    for ch in ln:
        if ch == '\r':
            break
        line += ch
    line = line.replace("v2", "")
    text.append(line)
data['text'] = text

# Rename columns for clarity
data.columns = ["id", "label", "text", "label_num"]
print(data.head())

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.3, random_state=42)

# Vectorize text
vectorizer = CountVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(x_train_vec, y_train)

# Predict
y_pred = model.predict(x_test_vec)

# Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))
