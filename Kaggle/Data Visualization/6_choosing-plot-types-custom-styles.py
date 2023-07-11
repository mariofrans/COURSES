import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.plotting.register_matplotlib_converters()

path_spotify = 'Jobs/Kaggle/Data Visualization/input/spotify.csv'

##################################################################################################################

""" CHOOSING PLOT TYPES & CUSTOM STYLES """

spotify_data = pd.read_csv(path_spotify, index_col="Date", parse_dates=True)

# list of available themes:
# "darkgrid"
# "whitegrid"
# "dark"
# "white"
# "ticks"

# Change the style of the figure
sns.set_style("dark")

# Line chart 
plt.figure(figsize=(12,6))
sns.lineplot(data=spotify_data)

##################################################################################################################