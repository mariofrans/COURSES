import pandas as pd

# read file content
filename_150 = 'Jobs/Kaggle/Pandas/input/winemag-data-150k.csv'
reviews = pd.read_csv(filename_150, index_col=[0])

##################################################################################################################

""" DATA TYPES & MISSING VALUES """

print("\n===============================================================\n")

dtype = reviews['points'].dtype
print("Datatype of Column:", dtype)

print("\n===============================================================\n")

point_strings = pd.Series(list(reviews['points'])).astype(str)
print("Series of Points Column (Converted to String Datatype):")
print(point_strings)

print("\n===============================================================\n")

# number of reviews in the dataset containing a missing price
n_missing_prices = len(reviews[pd.isnull(reviews['price'])])
print("Number of Reviews in Dataset Containing a Missing Price:", n_missing_prices)

print("\n===============================================================\n")

# series counting the number of times each value occurs in the 'region_1' column
# sort in descending order
reviews_per_region = reviews['region_1'].fillna("Unknown").value_counts()
reviews_per_region = reviews_per_region.sort_values(ascending=False)
print("Series of Points Column (Converted to String Datatype):")
print(reviews_per_region)

print("\n===============================================================\n")

##################################################################################################################