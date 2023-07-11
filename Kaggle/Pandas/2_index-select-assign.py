import pandas as pd

# read file content
filename_150 = 'Jobs/Kaggle/Pandas/input/winemag-data-150k.csv'
reviews = pd.read_csv(filename_150, index_col=[0])

##################################################################################################################

""" INDEXING, SELECTING & ASSIGNING"""

print("\n===============================================================\n")

country_column = reviews['country']
print("Select 'country' Column Only:")
print(country_column)

print("\n===============================================================\n")

# first country in selected column
first_country = country_column[0]
print("Select First Country in 'country' Column:", first_country)

print("\n===============================================================\n")

# first few countries in column
first_few_data = country_column.iloc[0:3]
print("Select First Few Countries in 'country' Column:")
print(first_few_data)

print("\n===============================================================\n")

# first row in dataframe, store them in a series
first_row = reviews.iloc[0]
print("Select First Row Only:")
print(first_row)

print("\n===============================================================\n")

# select rows by index
indexes = [2, 4, 9, 10, 11, 12, 15, 30, 1000, 1005, 2000, 2002, 2019]
filtered_rows = reviews.iloc[indexes]
print("Filter by Selected Rows Only:")
print(filtered_rows)

print("\n===============================================================\n")

# find by column names & rows indexes
df_country_variety = reviews.loc[0:3, ['country', 'region_2']]
print('Select Custom Rows & Columns:')
print(df_country_variety)

print("\n===============================================================\n")

# filter data by country name
italian_wines = reviews.loc[reviews['country']=='Italy']
print('Select Italy Data Only:')
print(italian_wines)

print("\n===============================================================\n")

# select column & rows with multiple conditions
top_us_italy_wines = reviews.loc[ (reviews['country'].isin(['US', 'Italy'])) & (reviews['points'] >= 95) ]
print('Select Custom Columns With Conditions:')
print(top_us_italy_wines)

print("\n===============================================================\n")

##################################################################################################################