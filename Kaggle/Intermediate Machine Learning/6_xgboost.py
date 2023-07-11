import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Path of the file to be read
path_train_data = 'Jobs/Kaggle/Intermediate Machine Learning/input/train.csv'
path_test_data = 'Jobs/Kaggle/Intermediate Machine Learning/input/test.csv'
path_submission = 'Jobs/Kaggle/Intermediate Machine Learning/output/submission.csv'

# Read the data
train_data = pd.read_csv(path_train_data, index_col='Id')
test_data = pd.read_csv(path_test_data, index_col='Id')

##################################################################################################################

# remove rows with missing target
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data['SalePrice']            

# drop 'SalePrice' column to separate target (y) from predictors
# X will be all train_data without the 'SalePrice' column
train_data.drop(['SalePrice'], axis=1, inplace=True)
X = train_data
X_test_full = test_data

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

##################################################################################################################

""" STEP 1: BUILD MODEL """

my_model_1 = XGBRegressor()
my_model_1.fit(X_train, y_train)
predictions_1 = my_model_1.predict(X_valid)

mae_1 = mean_absolute_error(y_valid, predictions_1)
print("Mean Absolute Error:" , mae_1)

##################################################################################################################

""" STEP 2: IMPROVE THE MODEL """

my_model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model_2.fit(X_train, y_train)
predictions_2 = my_model_2.predict(X_valid)

mae_2 = mean_absolute_error(y_valid, predictions_2)
print("Mean Absolute Error:" , mae_2)

##################################################################################################################

""" STEP 3: BREAK THE MODEL """

my_model_3 = XGBRegressor(n_estimators=10, learning_rate=0.05)
my_model_3.fit(X_train, y_train)
predictions_3 = my_model_3.predict(X_valid)

mae_3 = mean_absolute_error(y_valid, predictions_3)
print("Mean Absolute Error:" , mae_3)

##################################################################################################################
