import pandas as pd

# read file content
filename_150 = 'Jobs/Kaggle/Pandas/input/winemag-data-150k.csv'
filename_gaming = 'Jobs/Kaggle/Pandas/input/gaming.csv'
filename_movies = 'Jobs/Kaggle/Pandas/input/movies.csv'
filename_meets = 'Jobs/Kaggle/Pandas/input/meets.csv'
filename_openpowerlifting = 'Jobs/Kaggle/Pandas/input/openpowerlifting.csv'

reviews = pd.read_csv(filename_150, index_col=[0])
gaming_products = pd.read_csv(filename_gaming, index_col=[0])
movie_products = pd.read_csv(filename_movies, index_col=[0])
powerlifting_meets = pd.read_csv(filename_meets, index_col=[0])
powerlifting_competitors = pd.read_csv(filename_openpowerlifting, index_col=[0])

##################################################################################################################

""" RENAMING & COMBINING """

print("\n===============================================================\n")

# 'region_1' and 'region_2' columns renamed to 'region' and 'locale'
my_dictionary = {'region_1': 'region', 'region_2': 'locale'}
renamed_column_reviews = reviews.rename(columns=my_dictionary)
print("Renamed Columns:")
print(renamed_column_reviews.loc[ : , ['region', 'locale'] ])

print("\n===============================================================\n")

# rename indexes in the dataset to wines
renamed_index_reviews = reviews.rename_axis("wines", axis='rows')
print("Renamed Indexes:")
print(renamed_index_reviews)

print("\n===============================================================\n")

# concatenate all products mentioned on either subreddit (movies.csv or gaming.csv)
combined_products = pd.concat([gaming_products, movie_products])
print("Concatenated DataFrames (Movies + Gaming):")
print(combined_products)

print("\n===============================================================\n")

# Both tables include references to a MeetID, a unique key for each meet (competition) 
# included in the database. Using this, generate a dataset combining the two tables into one
powerlifting_meets = powerlifting_meets.set_index("MeetID")
powerlifting_competitors = powerlifting_competitors.set_index("MeetID")
powerlifting_combined = powerlifting_meets.join(powerlifting_competitors)

print("\n===============================================================\n")

##################################################################################################################