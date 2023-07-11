import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.plotting.register_matplotlib_converters()

path_ign = 'Jobs/Kaggle/Data Visualization/input/ign_scores.csv'

##################################################################################################################

""" BAR CHARTS & HEATMAPS """

##################################################################################################################

""" STEP 1: LOAD THE DATA """

ign_data = pd.read_csv(path_ign, index_col='Platform')

##################################################################################################################

""" STEP 2: REVIEW THE DATA """

print(ign_data)

# What is the highest average score received by PC games, for any genre?
high_score = max(list(ign_data.loc['PC']))
# print(high_score)

# On the Playstation Vita platform, which genre has the lowest average score? Please provide the name of 
# the column, and put your answer in single quotes (e.g., 'Action', 'Adventure', 'Fighting', etc.)
ps_vita_dict = dict(ign_data.loc['PlayStation Vita'])
worst_genre = min(ps_vita_dict, key=ps_vita_dict.get)
print(worst_genre)

##################################################################################################################

""" STEP 3: WHICH PLATFORM IS THE BEST? """

"""
Create a bar chart that shows the average score for racing games, for each platform. 
Your chart should have one bar for each platform.
"""

# Bar chart showing average score for racing games by platform
sns.barplot(x=ign_data.index, y=ign_data['Racing'])

"""
Based on the bar chart, do you expect a racing game for the Wii platform to receive a high rating? If not, what 
gaming platform seems to be the best alternative?
"""

"""
Based on the data, we should not expect a racing game for the Wii platform to receive a high rating. In fact, 
on average, racing games for Wii score lower than any other platform. Xbox One seems to be the best alternative, 
since it has the highest average ratings.
"""

##################################################################################################################

""" STEP 4: ALL POSSIBLE COMBINATIONS """

# Heatmap showing average game score by platform and genre
plt.figure(figsize=(10,10))
sns.heatmap(ign_data, annot=True)
plt.xlabel("Genre")
plt.title("Average Game Score, by Platform and Genre")

"""
Which combination of genre and platform receives the highest average ratings? Which combination receives the 
lowest average rankings?
"""

"""
Simulation games for Playstation 4 receive the highest average ratings (9.2). Shooting and Fighting games for 
Game Boy Color receive the lowest average rankings (4.5).
"""

##################################################################################################################