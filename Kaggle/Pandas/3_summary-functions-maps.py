import pandas as pd

# read file content
filename_150 = 'Jobs/Kaggle/Pandas/input/winemag-data-150k.csv'
reviews = pd.read_csv(filename_150, index_col=[0])

##################################################################################################################

""" SUMMARY FUNCTIONS & MAPS """

print("\n===============================================================\n")

median_points = reviews['points'].median()
print('Median of Selected Column:', median_points)

print("\n===============================================================\n")

# remove duplicates from column, store them in a list
countries = list(dict.fromkeys(list(reviews['country'])))
print('Remove Duplicates of Countries:')
print(countries)

print("\n===============================================================\n")

# count the reviews for each country, store them in a series
reviews_per_country = reviews['country'].value_counts()
print('Count Number of Reviews for Each Country:')
print(reviews_per_country)

print("\n===============================================================\n")

# calculate centered price
centered_price = reviews['price'] - reviews['price'].mean()
print("Calculate Centered Price for 'price' Column:")
print(centered_price)

print("\n===============================================================\n")

# find the best bargain price of wine (highest reviews to cost ratio)
# find the max index of the selected condition for dataframe
bargain_idx = (reviews['points'] / reviews['price']).idxmax()
bargain_wine = reviews.loc[bargain_idx]
print("The Most Bargain Wine is:")
print(bargain_wine)

print("\n===============================================================\n")

# count the number of 'tropical' and 'fruity' mentions in 'description' column, store them in a Series
n_trop = reviews['description'].map(lambda desc: "tropical" in desc).sum()
n_fruity = reviews['description'].map(lambda desc: "fruity" in desc).sum()
descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])
print("Number of 'tropical' & 'fruity' Mentions in 'description' Column:")
print(descriptor_counts)

print("\n===============================================================\n")

# create a star rating system based on the following conditions:
# Country = Canada --> 3 stars
# points>=95 --> 3 stars
# points>=85 --> 2 stars
# points<85 --> 1 star

def stars(row):
    if row.country == 'Canada': return 3
    elif row.points >= 95: return 3
    elif row.points >= 85: return 2
    else: return 1

star_ratings = reviews.apply(stars, axis='columns')
print("Star Ratings of Wines:")
print(star_ratings)

print("\n===============================================================\n")

##################################################################################################################