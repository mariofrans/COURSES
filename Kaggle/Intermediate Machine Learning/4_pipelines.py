import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Path of the file to be read
path_train_data = 'Jobs/Kaggle/Intermediate Machine Learning/input/train.csv'
path_test_data = 'Jobs/Kaggle/Intermediate Machine Learning/input/test.csv'
path_submission = 'Jobs/Kaggle/Intermediate Machine Learning/output/submission.csv'

# Read the data
train_data = pd.read_csv(path_train_data, index_col='Id')
test_data = pd.read_csv(path_test_data, index_col='Id')

##################################################################################################################

""" PIPELINES """

# remove rows with missing target
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data['SalePrice']

# drop 'SalePrice' column to separate target (y) from predictors
# X will be all train_data without the 'SalePrice' column
train_data.drop(['SalePrice'], axis=1, inplace=True)
X = train_data
X_test = test_data

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(train_data, y, train_size=0.8, test_size=0.2, random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = test_data[my_cols].copy()
X_train.head()

##################################################################################################################

# preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
# bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# define model
model = RandomForestRegressor(n_estimators=100, random_state=0)
# bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])
# preprocessing of training data, fit model 
clf.fit(X_train, y_train)
# preprocessing of validation data, get predictions
preds = clf.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))

##################################################################################################################

""" STEP 1: IMPROVE THE PERFORMANCE """

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)
# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)

##################################################################################################################

""" STEP 2: GENERATE TEST PREDICTIONS """

# Preprocessing of test data, fit model
preds_test = my_pipeline.predict(X_test)

# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
output.to_csv(path_submission, index=False)

##################################################################################################################