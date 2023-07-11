import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

path_df = "Jobs/Kaggle/Feature Engineering/input/ames.csv"

##################################################################################################################

""" MUTUAL INFORMATION """

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=14, titlepad=10, )

# Utility functions from Tutorial
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]): 
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

df = pd.read_csv(path_df)
features = ["YearBuilt", "MoSold", "ScreenPorch"]
sns.relplot(x="value", y="SalePrice", col="variable", data=df.melt(id_vars="SalePrice", 
            value_vars=features), facet_kws=dict(sharex=False), )

##################################################################################################################

""" STEP 1: UNDERSTAND MUTUAL INFORMATION """

"""
Based on the plots, which feature do you think would have the highest mutual information with SalePrice?
"""

"""
Based on the plots, YearBuilt should have the highest MI score since knowing the year tends to constrain 
SalePrice to a smaller range of possible values. This is generally not the case for MoSold, however. Finally, 
since ScreenPorch is usually just one value, 0, on average it won't tell you much about SalePrice (though more 
than MoSold).
"""

"""
The Ames dataset has seventy-eight features -- a lot to work with all at once! Fortunately, you can identify 
the features with the most potential.

Use the make_mi_scores function (introduced in the tutorial) to compute mutual information scores for the Ames 
features:
"""

X = df.copy()
y = X.pop('SalePrice')

mi_scores = make_mi_scores(X, y)
plt.figure(dpi=100, figsize=(8, 5))

"""
Now examine the scores using the functions in this cell. Look especially at top and bottom ranks.
"""

# uncomment to see top 20
print(mi_scores.head(20))
# uncomment to see bottom 20
print(mi_scores.tail(20))  

# uncomment to see top 20
plot_mi_scores(mi_scores.head(20))
# uncomment to see bottom 20
plot_mi_scores(mi_scores.tail(20))

##################################################################################################################

""" STEP 2: EXAMINE MI SCORES """

"""
Do the scores seem reasonable? Do the high scoring features represent things you'd think most people would 
value in a home? Do you notice any themes in what they describe?
"""

"""
Some common themes among most of these features are:
    1. Location: Neighborhood
    2. Size: all of the Area and SF features, and counts like FullBath and GarageCars
    3. Quality: all of the Qual features
    4. Year: YearBuilt and YearRemodAdd
    5. Types: descriptions of features and styles like Foundation and GarageType

These are all the kinds of features you'll commonly see in real-estate listings (like on Zillow), It's good 
then that our mutual information metric scored them highly. On the other hand, the lowest ranked features seem 
to mostly represent things that are rare or exceptional in some way, and so wouldn't be relevant to the average 
home buyer.
"""

"""
In this step you'll investigate possible interaction effects for the BldgType feature. This feature describes 
the broad structure of the dwelling in five categories:

Bldg Type (Nominal):    Type of dwelling
1Fam                    Single-family Detached    
2FmCon                  Two-family Conversion; originally built as one-family dwelling
Duplx                   Duplex
TwnhsE                  Townhouse End Unit
TwnhsI                  Townhouse Inside Unit`

The BldgType feature didn't get a very high MI score. A plot confirms that the categories in BldgType don't 
do a good job of distinguishing values in SalePrice (the distributions look fairly similar, in other words):
"""

sns.catplot(x="BldgType", y="SalePrice", data=df, kind="boxen")

"""
Still, the type of a dwelling seems like it should be important information. Investigate whether BldgType 
produces a significant interaction with either of the following:
    1. GrLivArea  # Above ground living area
    2. MoSold     # Month sold

Run the following cell twice, the first time with feature = "GrLivArea" and the next time with feature="MoSold":
"""

# feature = "GrLivArea"
feature = "MoSold"

sns.lmplot( x=feature, y="SalePrice", hue="BldgType", col="BldgType", data=df, 
            scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=4, )

"""
The trend lines being significantly different from one category to the next indicates an interaction effect.
"""

##################################################################################################################

""" STEP 3: DISCOVER INTERACTIONS """

"""
From the plots, does BldgType seem to exhibit an interaction effect with either GrLivArea or MoSold?
"""

"""
The trends lines within each category of BldgType are clearly very different, indicating an interaction between 
these features. Since knowing BldgType tells us more about how GrLivArea relates to SalePrice, we should 
consider including BldgType in our feature set.

The trend lines for MoSold, however, are almost all the same. This feature hasn't become more informative for 
knowing BldgType
"""

##################################################################################################################