import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt

path_earthquakes = 'Jobs/Kaggle/Data Cleaning/input/database.csv'
earthquakes = pd.read_csv(path_earthquakes)

##################################################################################################################

""" PARSING DATES """

##################################################################################################################

""" STEP 1: CHECK THE DATATYPE OF OUR DATE COLUMN """

# prints object datatype
print(earthquakes['Date'].dtype)

##################################################################################################################

""" STEP 2: CONVERT DATE COLUMN INTO DATETIME FORMAT """

"""
Most of the entries in the "Date" column follow the same format: "month/day/four-digit year". However, 
the entry at index 3378 follows a completely different pattern. Run the code cell below to see this.
"""

print(earthquakes[3378:3383])

"""
This does appear to be an issue with data entry: ideally, all entries in the column have the same format. 
We can get an idea of how widespread this issue is by checking the length of each entry in the "Date" column.
"""

date_lengths = earthquakes.Date.str.len()
print(date_lengths.value_counts())

"""
Looks like there are two more rows that has a date in a different format. Run the code cell below to obtain 
the indices corresponding to those rows and print the data.
"""

indices = np.where([date_lengths == 24])[1]
print('Indices with corrupted data:', indices)
print(earthquakes.loc[indices])

"""
Given all of this information, it's your turn to create a new column "date_parsed" in the earthquakes dataset 
that has correctly parsed dates in it.
"""

# fix the corrupt data manually
earthquakes.loc[3378, "Date"] = "02/23/1975"
earthquakes.loc[7512, "Date"] = "04/28/1985"
earthquakes.loc[20650, "Date"] = "03/13/2011"
# reformat the fixed data
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format="%m/%d/%Y")

##################################################################################################################

""" STEP 3: SELECT THE DAY OF THE MONTH """

day_of_month_earthquakes = earthquakes['date_parsed'].dt.day

##################################################################################################################

""" STEP 4: PLOT THE DAY OF MONTH TO CHECK THE DATE PARSING """

sns.distplot(day_of_month_earthquakes, kde=False, bins=31)

"""
The graph should make sense: it shows a relatively even distribution in days of the month,which is what we 
would expect.
"""

##################################################################################################################