import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from category_encoders import MEstimateEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

path_df = "Jobs/Kaggle/Feature Engineering/input/ames.csv"

##################################################################################################################

""" TARGET ENCODING """

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=14, titlepad=10, )
warnings.filterwarnings('ignore')

def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_log_error", )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

df = pd.read_csv(path_df)

"""
First you'll need to choose which features you want to apply a target encoding to. Categorical features with a 
large number of categories are often good candidates. Run this cell to see how many categories each categorical 
feature in the Ames dataset has.
"""

print(df.select_dtypes(["object"]).nunique())

"""
We talked about how the M-estimate encoding uses smoothing to improve estimates for rare categories. To see how 
many times a category occurs in the dataset, you can use the value_counts method. This cell shows the counts for 
SaleType, but you might want to consider others as well.
"""

print(df["SaleType"].value_counts())

##################################################################################################################

""" STEP 1: CHOOSE FEATURES FOR ENCODING """

"""
Which features did you identify for target encoding?
"""

"""
The Neighborhood feature looks promising. It has the most categories of any feature, and several categories 
are rare. Others that could be worth considering are SaleType, MSSubClass, Exterior1st, Exterior2nd. In fact, 
almost any of the nominal features would be worth trying because of the prevalence of rare categories.
"""

# Encoding split
X_encode = df.sample(frac=0.20, random_state=0)
y_encode = X_encode.pop("SalePrice")

# Training split
X_pretrain = df.drop(X_encode.index)
y_train = X_pretrain.pop("SalePrice")

##################################################################################################################

""" STEP 2: APPLY M-ESTIMATE ENCODING """

# YCreate the MEstimateEncoder
# Choose a set of features to encode and a value for m
encoder = MEstimateEncoder(cols=["Neighborhood"], m=2.0)
# Fit the encoder on the encoding split
encoder.fit(X_encode, y_encode)
# Encode the training split
X_train = encoder.transform(X_pretrain, y_train)

"""
See how the encoded feature compares to the target
"""

feature = encoder.cols

plt.figure(dpi=90)
ax = sns.distplot(y_train, kde=True, hist=False)
ax = sns.distplot(X_train[feature], color='r', ax=ax, hist=True, kde=False, norm_hist=True)
ax.set_xlabel("SalePrice")

"""
From the distribution plots, does it seem like the encoding is informative?
Score of the encoded set compared to the original set:
"""

X = df.copy()
y = X.pop("SalePrice")
score_base = score_dataset(X, y)
score_new = score_dataset(X_train, y_train)

print(f"Baseline Score: {score_base:.4f} RMSLE")
print(f"Score with Encoding: {score_new:.4f} RMSLE")

"""
Do you think that target encoding was worthwhile in this case? Depending on which feature or features you 
chose, you may have ended up with a score significantly worse than the baseline. In that case, it's likely 
the extra information gained by the encoding couldn't make up for the loss of data used for the encoding.
"""

# Try experimenting with the smoothing parameter m = 0, 1, 5, 50
m = 0
# m = 1
# m = 5
# m = 50

X = df.copy()
y = X.pop('SalePrice')

# Create an uninformative feature
X["Count"] = range(len(X))
# actually need one duplicate value to circumvent error-checking in MEstimateEncoder
X["Count"][1] = 0  

# fit and transform on the same dataset
encoder = MEstimateEncoder(cols="Count", m=m)
X = encoder.fit_transform(X, y)

# Results
score =  score_dataset(X, y)
print(f"Score: {score:.4f} RMSLE")

plt.figure(dpi=90)
ax = sns.distplot(y, kde=True, hist=False)
ax = sns.distplot(X["Count"], color='r', ax=ax, hist=True, kde=False, norm_hist=True)
ax.set_xlabel("SalePrice")

##################################################################################################################

""" STEP 3: OVERFITTING THE TARGET ENCODERS """

"""
Based on your understanding of how mean-encoding works, can you explain how XGBoost was able to get an almost a 
perfect fit after mean-encoding the count feature?
"""

"""
Since Count never has any duplicate values, the mean-encoded Count is essentially an exact copy of the target. 
In other words, mean-encoding turned a completely meaningless feature into a perfect feature.

Now, the only reason this worked is because we trained XGBoost on the same set we used to train the encoder. 
If we had used a hold-out set instead, none of this "fake" encoding would have transferred to the training data.

The lesson is that when using a target encoder it's very important to use separate data sets for training the 
encoder and training the model. Otherwise the results can be very disappointing!
"""

##################################################################################################################