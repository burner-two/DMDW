# Integrate data from multiple sources by merging and transforming datasets using Python's pandas library and data manipulation techniques.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV files
df1 = pd.read_csv('Student.csv')
df2 = pd.read_csv('Marks.csv')

# Show the first few rows
print("Student Data:")
print(df1.head())

print("\nMarks Data:")
print(df2.head())

# Merge the dataframes on 'StudentID'
df = pd.merge(df1, df2, on="StudentID")

# Display merged dataframe
print("\nMerged DataFrame:")
print(df.head(10))

# Sort by Marks
sorted_df = df.sort_values(by=['Marks'])
print("\nSorted by Marks:")
print(sorted_df)

# Filter specific columns
filtered_df = df[['Class', 'Subject', 'Marks']]
print("\nFiltered Columns (Class, Subject, Marks):")
print(filtered_df)

# Check for duplicates
print("\nDuplicate Rows (True = duplicate):")
print(df.duplicated())

# Remove duplicates
df_no_duplicates = df.drop_duplicates()
print("\nData after removing duplicates:")
print(df_no_duplicates)

# Rename the 'Name' column to 'StudentName'
df_renamed = df.rename(columns={'Name': 'StudentName'})
print("\nDataFrame with Renamed Column:")
print(df_renamed)
