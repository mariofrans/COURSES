import pandas as pd
from collections import defaultdict
import spacy
from spacy.matcher import PhraseMatcher

path_data = 'Jobs/Kaggle/Natural Language Processing/input/restaurant.json'
data = pd.read_json(path_data)
# print(data.head())

##################################################################################################################

""" INTRO TO NLP """

"""
You're a consultant for DelFalco's Italian Restaurant. The owner asked you to identify whether there are any 
foods on their menu that diners find disappointing.

The business owner suggested you use diner reviews from the Yelp website to determine which dishes people liked 
and disliked. You pulled the data from Yelp. Before you get to analysis, run the code cell below for a quick 
look at the data you have to work with.

The owner also gave you this list of menu items and common alternate spellings...
"""

menu = ["Cheese Steak", "Cheesesteak", "Steak and Cheese", "Italian Combo", "Tiramisu", "Cannoli",
        "Chicken Salad", "Chicken Spinach Salad", "Meatball", "Pizza", "Pizzas", "Spaghetti",
        "Bruchetta", "Eggplant", "Italian Beef", "Purista", "Pasta", "Calzones",  "Calzone",
        "Italian Sausage", "Chicken Cutlet", "Chicken Parm", "Chicken Parmesan", "Gnocchi",
        "Chicken Pesto", "Turkey Sandwich", "Turkey Breast", "Ziti", "Portobello", "Reuben",
        "Mozzarella Caprese",  "Corned Beef", "Garlic Bread", "Pastrami", "Roast Beef",
        "Tuna Salad", "Lasagna", "Artichoke Salad", "Fettuccini Alfredo", "Chicken Parmigiana",
        "Grilled Veggie", "Grilled Veggies", "Grilled Vegetable", "Mac and Cheese", "Macaroni",  
         "Prosciutto", "Salami"]

##################################################################################################################

""" STEP 1: PLAN YOUR ANALYSIS """

"""
Given the data from Yelp and the list of menu items, do you have any ideas for how you could find which menu 
items have disappointed diners?

Think about your answer. Then run the cell below to see one approach
"""

"""
You could group reviews by what menu items they mention, and then calculate the average rating for reviews that 
mentioned each item. You can tell which foods are mentioned in reviews with low scores, so the restaurant can 
fix the recipe or remove those foods from the menu.
"""

##################################################################################################################

""" STEP 2: FIND ITEMS IN ONE REVIEW """

"""
You'll pursue this plan of calculating average scores of the reviews mentioning each menu item.
As a first step, you'll write code to extract the foods mentioned in a single review.
Since menu items are multiple tokens long, you'll use PhraseMatcher which can match series of tokens
"""

index_of_review_to_test_on = 14
text_to_test_on = data.text.iloc[index_of_review_to_test_on]

# Load the SpaCy model
nlp = spacy.blank('en')

# Create the tokenized version of text_to_test_on
review_doc = nlp(text_to_test_on)

# Create the PhraseMatcher object. The tokenizer is the first argument. 
# Use attr = 'LOWER' to make consistent capitalization
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

# Create a list of tokens for each item in the menu
menu_tokens_list = [nlp(item) for item in menu]

matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
menu_tokens_list = [nlp(item) for item in menu]
matcher.add("MENU", menu_tokens_list)
matches = matcher(review_doc)

for match in matches:
   print(f"Token number {match[1]}: {review_doc[match[1]:match[2]]}")

# Output:
# Token number 2: Purista
# Token number 16: prosciutto
# Token number 58: meatball

##################################################################################################################

""" STEP 3: MATCHING ON THE WHOLE DATASET """

"""
Now run this matcher over the whole dataset and collect ratings for each menu item. Each review has a rating, 
review.stars. For each item that appears in the review text (review.text), append the review's rating to a list 
of ratings for that item. The lists are kept in a dictionary item_ratings.

To get the matched phrases, you can reference the PhraseMatcher documentation for the structure of each match 
object:
    A list of (match_id, start, end) tuples, describing the matches. A match tuple describes a span doc[start:end]. 
    The match_id is the ID of the added match pattern.
"""

# item_ratings is a dictionary of lists. If a key doesn't exist in item_ratings,
# the key is added with an empty list as the value.
item_ratings = defaultdict(list)

for idx, review in data.iterrows():
    doc = nlp(review.text)
    # Using the matcher from the previous exercise
    matches = matcher(doc)
    
    # Create a set of the items found in the review text
    found_items = set([doc[match[1]:match[2]].lower_ for match in matches])
    
    # Update item_ratings with rating for each item in found_items
    # Transform the item strings to lowercase to make it case insensitive
    for item in found_items: item_ratings[item].append(review.stars)

##################################################################################################################

""" STEP 4: WHAT'S THE WORST ITEM REVIEWED? """

"""
Using these item ratings, find the menu item with the worst average rating
"""

# Calculate the mean ratings for each menu item as a dictionary
mean_ratings = {item: sum(ratings)/len(ratings) for item, ratings in item_ratings.items()}

# Find the worst item, and write it as a string in worst_item. This can be multiple lines of code if you want.
worst_item = sorted(mean_ratings, key=mean_ratings.get)[0]

print(worst_item)
print(mean_ratings[worst_item])

##################################################################################################################

""" STEP 5: ARE COUNTS IMPORTANT HERE? """

"""
Similar to the mean ratings, you can calculate the number of reviews for each item
"""

counts = {item: len(ratings) for item, ratings in item_ratings.items()}

item_counts = sorted(counts, key=counts.get, reverse=True)
for item in item_counts: print(f"{item:>25}{counts[item]:>5}")

"""
Here is code to print the 10 best and 10 worst rated items. Look at the results, and decide whether you think 
it's important to consider the number of reviews when interpreting scores of which items are best and worst.
"""

sorted_ratings = sorted(mean_ratings, key=mean_ratings.get)

print("Worst rated menu items:")
for item in sorted_ratings[:10]: print(f"{item:20} Ave rating: {mean_ratings[item]:.2f} \tcount: {counts[item]}")
    
print("\n\nBest rated menu items:")
for item in sorted_ratings[-10:]: print(f"{item:20} Ave rating: {mean_ratings[item]:.2f} \tcount: {counts[item]}")

"""
The less data you have for any specific item, the less you can trust that the average rating is the "real" 
sentiment of the customers. This is fairly common sense. If more people tell you the same thing, you're more 
likely to believe it. It's also mathematically sound. As the number of data points increases, the error on 
the mean decreases as 1 / sqrt(n).
"""

##################################################################################################################