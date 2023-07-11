import pandas as pd
import numpy as np

# helpful modules
import fuzzywuzzy
from fuzzywuzzy import process
import chardet

path_professors = 'Jobs/Kaggle/Data Cleaning/input/pakistan_intellectual_capital.csv'
professors = pd.read_csv(path_professors)

##################################################################################################################

""" INCONSISTENT DATA ENTRY """

# convert to lower case
professors['Country'] = professors['Country'].str.lower()
# remove trailing white spaces
professors['Country'] = professors['Country'].str.strip()

# get the top 10 closest matches to "south korea"
countries = professors['Country'].unique()
matches = fuzzywuzzy.process.extract("south korea", countries, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

def replace_matches_in_column(df, column, string_to_match, min_ratio = 47):
    # get a list of unique strings
    strings = df[column].unique()
    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
    # only get matches with a ratio > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]
    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)
    # replace all rows with close matches with the input matches 
    df.loc[rows_with_matches, column] = string_to_match
    # let us know the function's done
    print("All done!")
    
replace_matches_in_column(df=professors, column='Country', string_to_match="south korea")
countries = professors['Country'].unique()

##################################################################################################################

""" STEP 1: EXAMINE ANOTHER COLUMN """

"""
Write code below to take a look at all the unique values in the "Graduated from" column.
"""

graduated = professors['Graduated from'].unique()
graduated.sort()
print(graduated)

"""
Do you notice any inconsistencies in the data? Can any of the inconsistencies in the data be fixed by removing 
white spaces at the beginning and end of cells?
"""

"""
There are inconsistencies that can be fixed by removing white spaces at the beginning and end of cells. 
For instance, "University of Central Florida" and " University of Central Florida" both appear in the column.
"""

##################################################################################################################

""" STEP 2: DO SOME TEXT-PREPROCESSING """

"""
Convert every entry in the "Graduated from" column in the professors DataFrame to remove white spaces at the 
beginning and end of cells.
"""

# remove white spaced in the beginning & end of data
professors['Graduated from'] = professors['Graduated from'].str.strip()

##################################################################################################################

""" STEP 3: CONTINUE WORKING WITH COUNTRIES """

# get all the unique values in the 'City' column
countries = professors['Country'].unique()

# sort them alphabetically and then take a closer look
countries.sort()
print(countries)

"""
Take another look at the "Country" column and see if there's any more data cleaning we need to do.

It looks like 'usa' and 'usofa' should be the same country. Correct the "Country" column in the dataframe 
so that 'usofa' appears instead as 'usa'.
"""

"""
Take another look at the "Country" column and see if there's any more data cleaning we need to do.

It looks like 'usa' and 'usofa' should be the same country. Correct the "Country" column in the dataframe 
so that 'usofa' appears instead as 'usa'.

Use the most recent version of the DataFrame (with the whitespaces at the beginning and end of cells removed) 
from question 2
"""

matches = fuzzywuzzy.process.extract("usa", countries, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
replace_matches_in_column(df=professors, column='Country', string_to_match="usa", min_ratio=70)

##################################################################################################################