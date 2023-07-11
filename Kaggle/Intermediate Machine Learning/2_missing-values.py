import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Path of the file to be read
path_train_data = 'Jobs/Kaggle/Intermediate Machine Learning/input/train.csv'
path_test_data = 'Jobs/Kaggle/Intermediate Machine Learning/input/test.csv'
path_submission = 'Jobs/Kaggle/Intermediate Machine Learning/output/submission.csv'

# Read the data
train_data = pd.read_csv(path_train_data, index_col='Id')
test_data = pd.read_csv(path_test_data, index_col='Id')

##################################################################################################################

""" MISSING VALUES """

# remove rows with missing target
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data['SalePrice']

# drop 'SalePrice' column to separate target (y) from predictors
# X will be all train_data without the 'SalePrice' column
train_data.drop(['SalePrice'], axis=1, inplace=True)
# To keep things simple, we'll use only numerical predictors
X = train_data.select_dtypes(exclude=['object'])
X_test = test_data.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
X_train.head()

##################################################################################################################

""" STEP 1: PRELIMINARY INVESTIGATION """

# Shape of training data (num_rows, num_columns)
print(X_train.shape)

# number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
# only select columns where missing values are present
missing_val_count_by_column_positive = missing_val_count_by_column[missing_val_count_by_column > 0]
print(missing_val_count_by_column_positive)

# rows are in the training data
num_rows = len(X_train)
# columns in the training data have missing values
num_cols_with_missing = len(missing_val_count_by_column_positive)
# missing entries are contained in all of the training data
tot_missing = sum(missing_val_count_by_column_positive)

##################################################################################################################

""" STEP 2: DROP COLUMNS WITH MISSING VALUES """

# get names of columns with missing values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
# drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

# function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# find MAE of filtered data
print("MAE (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

##################################################################################################################

""" STEP 3: INPUTATION """

# input data to empty sports in the data
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

# find MAE using inputation method
print("MAE (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

##################################################################################################################

""" STEP 4: GENERATE TEST PREDICTIONS """

# preprocessed training and validation features
final_imputer = SimpleImputer(strategy='median')
final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(final_imputer.transform(X_valid))

# imputation removed column names; put them back
final_X_train.columns = X_train.columns
final_X_valid.columns = X_valid.columns

# define and fit model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(final_X_train, y_train)

# get validation predictions and its MAE
preds_valid = model.predict(final_X_valid)
print("MAE (Your approach):")
print(mean_absolute_error(y_valid, preds_valid))

# preprocess test data
final_X_test = pd.DataFrame(final_imputer.transform(X_test))
# get test predictions
preds_test = model.predict(final_X_test)

# save test predictions to file
output = pd.DataFrame({'Id': X_test.index,'SalePrice': preds_test})
output.to_csv(path_submission, index=False)

##################################################################################################################