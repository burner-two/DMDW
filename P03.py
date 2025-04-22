#Apply feature selection techniques like variance thresholding and correlation analysis using python’s scikit-learn library to reduce dimensionality in a dataset.

import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# Load the breast cancer dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train , X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#fit LassoCV model
Lasso_cv = LassoCV(cv=5)
Lasso_cv.fit(X_train,y_train)

#feature selection

sfm = SelectFromModel(Lasso_cv,prefit=True)
X_train_selected = sfm.transform(X_train)
X_test_selected = sfm.transform(X_test)

#train a Random forest classifier using the selected feature 
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train_selected,y_train)

#evaluate the model 
y_pred = model.predict = (X_test_selected)
print(classification_report,(y_test,y_pred))

selected_feature_indicies = np.where(sfm.get_support())[0]
selected_feature = cancer.feature_names [selected_feature_indicies]
coefficient = Lasso_cv.coef_
print("Selected Feature :",selected_feature)
print("Coefficient feature :",coefficient)
