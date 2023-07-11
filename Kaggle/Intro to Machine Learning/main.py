import pandas as pd
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Path of the file to be read
path_train_data = 'Jobs/Kaggle/Intro to Machine Learning/input/train.csv'
path_test_data = 'Jobs/Kaggle/Intro to Machine Learning/input/test.csv'
path_submission = 'Jobs/Kaggle/Intro to Machine Learning/output/submission.csv'

train_data = pd.read_csv(path_train_data)
test_data = pd.read_csv(path_test_data)

##################################################################################################################

""" BASIC DATA EXPLORATION """

# # average lot size (rounded to nearest integer)?
# avg_lot_size = round(train_data['LotArea'].mean())

# # how old is the newest home (current year - the date in which it was built)
# current_year = int(datetime.today().strftime("%Y"))
# newest_home_year = int(train_data['YearBuilt'].max())
# newest_home_age = current_year - newest_home_year

##################################################################################################################

""" YOUR FIRST MACHINE LEARNING MODEL """

# print the list of columns in the dataset to find the name of the prediction target
print(train_data.columns)

y = train_data['SalePrice']
# Create the list of features below
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# Select data corresponding to features in feature_names
X = train_data[feature_names]

# review data
# print description or statistics from X
print(X.describe())
# print the top few lines
print(X.head())

# specify the model. 
# for model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=1)
# fit the model
iowa_model.fit(X, y)
# predict y_train data using X_train data
predictions = iowa_model.predict(X)
print(predictions)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(iowa_model.predict(X.head()))

##################################################################################################################

""" MODEL VALIDATION """

# train test split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

# Specify the model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit iowa_model with the training data
iowa_model.fit(train_X, train_y)

# predict with all validation observations
val_predictions = iowa_model.predict(val_X)
# print the top few validation predictions
print(val_predictions)
# print the top few actual prices from validation data
print(val_y)

# calculate the mean absolute error of the machine learning model
val_mae = mean_absolute_error(val_y, val_predictions)
print(val_mae)

##################################################################################################################

""" UNDERFITTING AND OVERFITTING """

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

max_leaf_nodes_list = [5, 25, 50, 100, 250, 500]
mae_list = []

# find the ideal tree size from candidate_max_leaf_nodes
for max_leaf_nodes in max_leaf_nodes_list:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    mae_list.append(my_mae)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

# Store the best value of max_leaf_nodes (5, 25, 50, 100, 250 or 500)
best_mae_index = mae_list.index(min(mae_list))
best_tree_size = max_leaf_nodes_list[best_mae_index]

# Fill in argument to make optimal size
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
# fit the final model
final_model.fit(X, y)
# predict final model for actual y_test data, using X_test data
X_test = test_data[feature_names]
final_model.predict(X_test)

##################################################################################################################

""" RANDOM FORESTS """

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
# fit your model
rf_model.fit(train_X, train_y)
preds = rf_model.predict(val_X)

# calculate the mean absolute error of your Random Forest model on the validation data
rf_val_mae = mean_absolute_error(val_y, preds)
print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))

##################################################################################################################

""" MACHINE LEARNING COMPETITIONS """

# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state = 1)
# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(X, y)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[feature_names]
# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X)

# Run the code to save predictions in the format used for competition scoring
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
output.to_csv(path_submission, index=False)

##################################################################################################################