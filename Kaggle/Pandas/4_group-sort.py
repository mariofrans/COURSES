import pandas as pd

# read file content
filename_130 = 'Jobs/Kaggle/Pandas/input/winemag-data-130k.csv'
reviews = pd.read_csv(filename_130, index_col=[0])

##################################################################################################################

""" GROUPING & SORTING """

print("\n===============================================================\n")

# count how many reviews each person wrote, by twitter username, store them in series
# grouping 'taster_twitter_handle' column by the same name, and their reviews count
# 'taster_twitter_handle' --> index
# reviews count --> values

reviews_written = reviews.groupby('taster_twitter_handle').size()
print("Reviewers & Number of Times They Review:")
print(reviews_written)

print("\n===============================================================\n")

# find the best wine 'points' that can be bought with amount of money (index)
# group 'price' column by the same prices, and find their max 'points'
# sort results by index
# 'price' --> index
# maximum points --> values

best_rating_per_price = reviews.groupby('price')['points'].max().sort_index()
print("Best Wine Points That Can Be Purchased Based On Price:")
print(best_rating_per_price)

print("\n===============================================================\n")

# the minimum and maximum prices for each variety of wine
price_extremes = reviews.groupby('variety')['price'].agg([min, max])
print("Minimum and Maximum Prices For Each Wine:")
print(price_extremes)

print("\n===============================================================\n")

# sorted in descending order based on minimum price, then on maximum price (to break ties)
sorted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending=False)
print("Sorted Minimum and Maximum Prices For Each Wine:")
print(sorted_varieties)

print("\n===============================================================\n")

# 'reviewers' --> index   
# average review score from reviewer --> values

reviewer_mean_ratings = reviews.groupby('taster_name')['points'].mean()
print("Average Review Score From Reviewer:")
print(reviewer_mean_ratings)
print(reviewer_mean_ratings.describe())

print("\n===============================================================\n")

# combination of countries and varieties that are the most common, store them in series
country_variety_counts = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)
print("Most Common Combination of Countries and Varieties:")
print(country_variety_counts)

print("\n===============================================================\n")

##################################################################################################################