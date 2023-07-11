import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.plotting.register_matplotlib_converters()

path_fifa = 'Jobs/Kaggle/Data Visualization/input/fifa.csv'

##################################################################################################################

""" HELLO SEABORN """

##################################################################################################################

""" STEP 1: EXPLORE THE FEEDBACK SYSTEM """

##################################################################################################################

""" STEP 2: LOAD THE DATA """

fifa_data = pd.read_csv(path_fifa, index_col="Date", parse_dates=True)

##################################################################################################################

""" STEP 3: PLOT THE DATA """

# Set the width and height of the figure
plt.figure(figsize=(16,6))
# Line chart showing how FIFA rankings evolved over time
sns.lineplot(data=fifa_data)

"""
Considering only the years represented in the dataset, which countries spent at least 5 consecutive years in 
the #1 ranked spot?
"""

"""
The only country that meets this criterion is Brazil (code: BRA), as it maintains the highest ranking in 
1996-2000. Other countries do spend some time in the number 1 spot, but Brazil is the only country that 
maintains it for at least five consecutive years
"""

##################################################################################################################
