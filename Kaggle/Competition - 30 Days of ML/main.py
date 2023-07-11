import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# remove pandas warnings
pd.set_option('mode.chained_assignment', None)

##################################################################################################################

path_train_data = 'Jobs/Kaggle/Competition - Insurance/input/train.csv'
path_test_data = 'Jobs/Kaggle/Competition - Insurance/input/test.csv'
path_submission = 'Jobs/Kaggle/Competition - Insurance/output/submission.csv'

# collect data from csv to pandas df
train_data = pd.read_csv(path_train_data)
test_data = pd.read_csv(path_test_data)
test_data_ratio = len(test_data)/(len(train_data) + len(test_data))

##################################################################################################################

""" SETTING UP TARGET & TRAINING DATA (FROM TRAIN DATASET ONLY) """

# select columns to be used for training (that potentially effects the results)
features = [col for col in train_data.columns if col not in ["id", "target"]]
# select categorical columns only
features_categorical = [col for col in features if "cat" in col]
# print(features)
# print(features_categorical)

X = train_data[features]
y = train_data["target"]
# print(X)
# print(y)

##################################################################################################################

""" PREPROCESSING DATA (ENCODING) """

# encode all categorical columns' data
ordinal_encoder = OrdinalEncoder()
X[features_categorical] = ordinal_encoder.fit_transform(X[features_categorical])
# print(X)

##################################################################################################################

""" MODEL VALIDATION """

# create split test data
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=test_data_ratio)

# create machine learning model
model = RandomForestRegressor()
model.fit(train_X, train_y)
val_predictions = model.predict(val_X)

val_mse = mean_squared_error(val_y, val_predictions)
print(val_mse)

##################################################################################################################
