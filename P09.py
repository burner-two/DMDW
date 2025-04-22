#Aim: Implement the Apriori algorithm in Python to mine frequent itemset from a retail transaction dataset and extract association rules.

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori , association_rules

data = pd.read_csv('online-retail-datasets.csv')
print(data)

data.Country.unique()

# Cleaning the data
data['Description'] = data['Description'].str.strip()
data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
data['InvoiceNo'] = data['InvoiceNo'].astype('str')
data = data[~data['InvoiceNo'].str.contains('C')]

# Grouping data by country and preparing baskets
basket_France = (data[data['Country'] == "France"]
    .groupby(['InvoiceNo', 'Description'])['Quantity']
    .sum().unstack().reset_index().fillna(0)
    .set_index('InvoiceNo'))

basket_UK = (data[data['Country'] == "United Kingdom"]
    .groupby(['InvoiceNo', 'Description'])['Quantity']
    .sum().unstack().reset_index().fillna(0)
    .set_index('InvoiceNo'))

basket_Por = (data[data['Country'] == "Portugal"]
    .groupby(['InvoiceNo', 'Description'])['Quantity']
    .sum().unstack().reset_index().fillna(0)
    .set_index('InvoiceNo'))

basket_Sweden = (data[data['Country'] == "Sweden"]
    .groupby(['InvoiceNo', 'Description'])['Quantity']
    .sum().unstack().reset_index().fillna(0)
    .set_index('InvoiceNo'))
def hot_encode(X):  # Use the correct argument name here
    if X <= 0:       # Access the argument using X
        return 0
    if X >= 1:       # Access the argument using X
        return 1

basket_encoded = basket_France.applymap(hot_encode)
basket_France = basket_encoded

basket_encoded = basket_UK.applymap(hot_encode)
basket_UK = basket_encoded

basket_encoded = basket_Por.applymap(hot_encode)
basket_Por = basket_encoded

basket_encoded = basket_Sweden.applymap(hot_encode)
basket_Sweden = basket_encoded
frq_items = apriori(basket_France, min_support = 0.05, use_colnames = True)
rules = association_rules(frq_items, metric = "lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending = [False, False])
print(rules.head())




################################################################################################3

# Aim: Implement the Apriori algorithm in Python to mine frequent itemsets 
# from a retail transaction dataset and extract association rules.

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
data = pd.read_csv('online-retail-datasets.csv')
print(data.head())

# Show unique countries
print(data['Country'].unique())

# ----------------------- Data Cleaning -----------------------
# Strip leading/trailing whitespaces from product descriptions
data['Description'] = data['Description'].str.strip()

# Drop rows with missing InvoiceNo
data.dropna(subset=['InvoiceNo'], inplace=True)

# Convert InvoiceNo to string and remove credit transactions (those containing 'C')
data['InvoiceNo'] = data['InvoiceNo'].astype(str)
data = data[~data['InvoiceNo'].str.contains('C')]

# ------------------ Helper Function for Encoding ------------------
def hot_encode(x):
    return 0 if x <= 0 else 1

# ----------------------- Country-wise Baskets -----------------------
def create_basket(data, country):
    basket = (data[data['Country'] == country]
              .groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))
    basket = basket.applymap(hot_encode)
    basket = basket.astype(bool)  # Fix deprecation warning
    return basket

basket_France = create_basket(data, 'France')
basket_UK = create_basket(data, 'United Kingdom')
basket_Por = create_basket(data, 'Portugal')
basket_Sweden = create_basket(data, 'Sweden')

# -------------------- Apply Apriori and Association Rules --------------------
# Example: France
frq_items = apriori(basket_France, min_support=0.05, use_colnames=True)

# Generate rules
rules = association_rules(frq_items, metric='lift', min_threshold=1)

# Sort rules by confidence and lift
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

# Display top 5 rules
print(rules.head())
