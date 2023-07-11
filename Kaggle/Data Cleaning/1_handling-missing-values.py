import pandas as pd
import numpy as np

path_sf_permits = 'Jobs/Kaggle/Data Cleaning/input/Building_Permits.csv'
sf_permits = pd.read_csv(path_sf_permits)

##################################################################################################################

""" HANDLING MISSING VALUES """

##################################################################################################################

""" STEP 1: TAKE A FIRST LOOK AT THE DATA """

print(sf_permits.head())
# print(sf_permits.shape) #(rows, columns)

##################################################################################################################

""" STEP 2: HOW MANY MISSING DATA POINTS DO WE HAVE? """

missing_data_per_column = sf_permits.isnull().sum()
total_cells = np.product(sf_permits.shape)
total_missing = missing_data_per_column.sum()

percent_missing = total_missing/total_cells *100
print(percent_missing, "%")

##################################################################################################################

""" STEP 3: FIGURE OUT WHY THE DATA IS MISSING """

"""
Look at the columns "Street Number Suffix" and "Zipcode" from the San Francisco Building Permits dataset. 
Both of these contain missing values.
    1. Which, if either, are missing because they don't exist?
    2. Which, if either, are missing because they weren't recorded?
"""

"""
If a value in the "Street Number Suffix" column is missing, it is likely because it does not exist. If a 
value in the "Zipcode" column is missing, it was not recorded.
"""

##################################################################################################################

""" STEP 4: DROP MISSING DATA: ROWS """

sf_permits_dropped_rows = sf_permits.dropna()
print(sf_permits)

##################################################################################################################

""" STEP 5: DROP MISSING DATA: COLUMNS """

sf_permits_with_na_dropped = sf_permits.dropna(axis=1)
dropped_columns = sf_permits.shape[1] - sf_permits_with_na_dropped.shape[1]
print(sf_permits_with_na_dropped)
print(dropped_columns)

##################################################################################################################

""" STEP 6: FILL MISSING VALUES AUTOMATICALLY """

# fill empty cells with the next data in its COLUMN
# then replace all the remaining NaN's with 0
sf_permits_with_na_imputed = sf_permits.fillna(method='bfill', axis=0).fillna(0)

##################################################################################################################

