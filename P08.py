import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
data = pd.read_csv('online-retail-dataset.csv')
print(data.head())
print(data.describe())

# Drop missing values
data.dropna(inplace=True)

# Fix column name typo
data.columns = data.columns.str.strip()  # clean whitespace
# Fixing 'UnitPrice' typo
data['Total_Amount'] = data['Quantity'] * data['UnitPrice']

# --- MONETARY ---
m = data.groupby('CustomerID')['Total_Amount'].sum().reset_index()

# --- FREQUENCY ---
f = data.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
f.columns = ['CustomerID', 'Frequency']

# --- RECENCY ---
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%m/%d/%Y %H:%M')
last_day = data['InvoiceDate'].max()
data['difference'] = (last_day - data['InvoiceDate']).dt.days
r = data.groupby('CustomerID')['difference'].min().reset_index()
r.columns = ['CustomerID', 'Recency']

# Combine RFM features
grouped_fd = pd.merge(m, f, on='CustomerID', how='inner')
RFM_df = pd.merge(grouped_fd, r, on='CustomerID', how='inner')
RFM_df.columns = ['CustomerID', 'Monetary', 'Frequency', 'Recency']

# --- OUTLIER REMOVAL ---
outlier_vars = ['Monetary', 'Recency', 'Frequency']
for column in outlier_vars:
    Q1 = RFM_df[column].quantile(0.25)
    Q3 = RFM_df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = RFM_df[(RFM_df[column] < lower) | (RFM_df[column] > upper)].index
    print(f"{len(outliers)} Outliers detected in column {column}")
    RFM_df.drop(outliers, inplace=True)

# --- SCALING ---
scaled_df = RFM_df[['Monetary', 'Frequency', 'Recency']]
scaler = StandardScaler()
RFM_df_scaled = scaler.fit_transform(scaled_df)
RFM_df_scaled = pd.DataFrame(RFM_df_scaled, columns=['Monetary', 'Frequency', 'Recency'])

# --- KMeans Clustering (fixed warning by setting n_init) ---
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(RFM_df_scaled)
RFM_df['labels'] = kmeans.labels_

# --- 3D VISUALIZATION ---
fig = plt.figure(figsize=(21, 10))
ax = fig.add_subplot(111, projection='3d')

for label in range(3):
    ax.scatter(RFM_df['Monetary'][RFM_df['labels'] == label],
               RFM_df['Frequency'][RFM_df['labels'] == label],
               RFM_df['Recency'][RFM_df['labels'] == label],
               label=f'Cluster {label}')

ax.set_xlabel('Monetary')
ax.set_ylabel('Frequency')
ax.set_zlabel('Recency')
ax.view_init(30, 185)
plt.legend()
plt.show()
# Boxplot for Monetary
plt.figure(figsize=(8, 5))
plt.boxplot(RFM_df['Monetary'], vert=False)
plt.title('Boxplot of Monetary Values')
plt.xlabel('Monetary')
plt.grid(True)
plt.show()
