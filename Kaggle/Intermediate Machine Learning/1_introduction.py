import pandas as pd
from sklearn.model_selection import train_test_split
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

""" INTRODUCTION """

# Obtain target and predictors
y = train_data['SalePrice']
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = train_data[features].copy()
X_test = test_data[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
X_train.head()

##################################################################################################################

""" STEP 1: EVALUATE SEVERAL MODELS """

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

# Function for comparing different models
def score_model(model):
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

models = [model_1, model_2, model_3, model_4, model_5]
mae_list = []
for model in models:
    mae = score_model(model)
    mae_list.append(mae)
    print("Model %d MAE: %d" % (models.index(model)+1, mae))

##################################################################################################################

""" STEP 2: GENERATE PREDICTIONS WITH THE BEST MODEL """

# Fill in the best model
best_model = models[mae_list.index(min(mae_list))]
# Define a model
my_model = best_model
# Fit the model to the training data
my_model.fit(X, y)
# Generate test predictions
preds_test = my_model.predict(X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
output.to_csv(path_submission, index=False)

##################################################################################################################