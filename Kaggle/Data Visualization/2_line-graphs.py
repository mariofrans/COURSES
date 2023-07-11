import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.plotting.register_matplotlib_converters()

path_museum = 'Jobs/Kaggle/Data Visualization/input/museum_visitors.csv'

##################################################################################################################

""" LINE GRAPHS """

##################################################################################################################

""" STEP 1: LOAD THE DATA """

museum_data = pd.read_csv(path_museum, index_col='Date', parse_dates=True)

##################################################################################################################

""" STEP 2: REVIEW THE DATA """

print(museum_data.tail())

# How many visitors did the Chinese American Museum receive in July 2018?
ca_museum_jul18 = 2620 
# In October 2018, how many more visitors did Avila Adobe receive than the Firehouse Museum?
avila_oct18 = 19280 - 4622

##################################################################################################################

""" STEP 3: CONVINCE THE MUSEUM BOARD """

# plot line graph
plt.figure(figsize=(12,6))
sns.lineplot(data=museum_data)

##################################################################################################################

""" STEP 4: ASSESS PERSONALITY """

# Line plot showing the number of visitors to Avila Adobe over time
plt.figure(figsize=(12,6))
sns.lineplot(data=museum_data['Avila Adobe'])

"""
Does Avila Adobe get more visitors:
    1. in September-February (in LA, the fall and winter months), or
    2. in March-August (in LA, the spring and summer)?

Using this information, when should the museum staff additional seasonal employees?
"""

"""
The line chart generally dips to relatively low values around the early part of each year (in December and January), 
and reaches its highest values in the middle of the year (especially around May and June). Thus, Avila Adobe 
usually gets more visitors in March-August (or the spring and summer months). With this in mind, Avila Adobe 
could definitely benefit from hiring more seasonal employees to help with the extra work in March-August 
(the spring and summer)!
"""

##################################################################################################################
