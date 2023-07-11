import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.plotting.register_matplotlib_converters()

path_ramen_ratings = 'Jobs/Kaggle/Data Visualization/input/ramen-ratings.csv'

##################################################################################################################

""" FINAL PROJECT """

my_data = pd.read_csv(path_ramen_ratings, index_col='date')
print(my_data.head())

# filter 'Unrated' data
my_data = my_data[my_data['Stars']!='Unrated']

# Create a plot
sns.distplot(my_data['Stars'], kde=False)

##################################################################################################################