#Build a decision tree classifier using python’s scikit learn library to predict customer churn based on historical data.


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

df = pd.read_csv('chrun.csv')
print(df)
df = df.dropna(['CustomerID','Age','Balance'],axis=1)
print(df)
