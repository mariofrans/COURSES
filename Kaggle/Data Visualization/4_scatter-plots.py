import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.plotting.register_matplotlib_converters()

path_candy = 'Jobs/Kaggle/Data Visualization/input/candy.csv'

##################################################################################################################

""" SCATTER PLOTS """

##################################################################################################################

""" STEP 1: LOAD THE DATA """

candy_data = pd.read_csv(path_candy, index_col='id')

##################################################################################################################

""" STEP 2: REVIEW THE DATA """

# Print the first five rows of the data
print(candy_data.head())

# Which candy was more popular with survey respondents: '3 Musketeers' or 'Almond Joy'?
winpercent_almond_joy = float(candy_data.loc[candy_data['competitorname']=='Almond Joy']['winpercent'])
winpercent_3_musketeers = float(candy_data.loc[candy_data['competitorname']=='3 Musketeers']['winpercent'])
# print(winpercent_almond_joy)
# print(winpercent_3_musketeers)

if winpercent_almond_joy>winpercent_3_musketeers: more_popular = 'Almond Joy'
elif winpercent_almond_joy<winpercent_3_musketeers: more_popular = '3 Musketeers'
else: print("Same Popularity")

# Which candy has higher sugar content: 'Air Heads' or 'Baby Ruth'?
sugarpercent_air_heads = float(candy_data.loc[candy_data['competitorname']=='Air Heads']['sugarpercent'])
sugarpercent_baby_ruth = float(candy_data.loc[candy_data['competitorname']=='Baby Ruth']['sugarpercent'])
# print(sugarpercent_air_heads)
# print(sugarpercent_baby_ruth)

if sugarpercent_air_heads>sugarpercent_baby_ruth: more_sugar = 'Air Heads'
elif sugarpercent_air_heads<sugarpercent_baby_ruth: more_sugar = 'Baby Ruth'
else: print("Same Sugar Count")

##################################################################################################################

""" STEP 3: THE ROLE OF SUGAR """

# Scatter plot showing the relationship between 'sugarpercent' and 'winpercent'
sns.scatterplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])

"""
Does the scatter plot show a strong correlation between the two variables? If so, are candies with more sugar 
relatively more or less popular with the survey respondents?
"""

"""
The scatter plot does not show a strong correlation between the two variables. Since there is no clear 
relationship between the two variables, this tells us that sugar content does not play a strong role in candy 
popularity.
"""

##################################################################################################################

""" STEP 4: TAKE A CLOSER LOOK """

# Scatter plot w/ regression line showing the relationship between 'sugarpercent' and 'winpercent'
sns.regplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])

"""
According to the plot above, is there a slight correlation between 'winpercent' and 'sugarpercent'? What does 
this tell you about the candy that people tend to prefer?
"""

"""
Since the regression line has a slightly positive slope, this tells us that there is a slightly positive 
correlation between 'winpercent' and 'sugarpercent'. Thus, people have a slight preference for candies containing 
relatively more sugar.
"""

##################################################################################################################

""" STEP 5: CHOCOLATE! """

# Scatter plot showing the relationship between 'pricepercent', 'winpercent', and 'chocolate'
sns.scatterplot(x=candy_data['pricepercent'], y=candy_data['winpercent'], hue=candy_data['chocolate'])

##################################################################################################################

""" STEP 6: INVESTIGATE CHOCOLATE """

# Color-coded scatter plot with regression lines
sns.lmplot(x="pricepercent", y="winpercent", hue="chocolate", data=candy_data)

"""
Using the regression lines, what conclusions can you draw about the effects of chocolate and price on candy 
popularity?
"""

"""
We'll begin with the regression line for chocolate candies. Since this line has a slightly positive slope, we 
can say that more expensive chocolate candies tend to be more popular (than relatively cheaper chocolate candies). 
Likewise, since the regression line for candies without chocolate has a negative slope, we can say that if 
candies don't contain chocolate, they tend to be more popular when they are cheaper. One important note, however, 
is that the dataset is quite small -- so we shouldn't invest too much trust in these patterns! To inspire more 
confidence in the results, we should add more candies to the dataset.
"""

##################################################################################################################

""" STEP 7: EVERYBODY LOVES CHOCOLATE """

# Scatter plot showing the relationship between 'chocolate' and 'winpercent'
sns.swarmplot(x=candy_data['chocolate'], y=candy_data['winpercent'])

"""
You decide to dedicate a section of your report to the fact that chocolate candies tend to be more popular than 
candies without chocolate. Which plot is more appropriate to tell this story: the plot from Step 6, or the plot 
from Step 7?
"""

"""
In this case, the categorical scatter plot from Step 7 is the more appropriate plot. While both plots tell the 
desired story, the plot from Step 6 conveys far more information that could distract from the main point.
"""

##################################################################################################################