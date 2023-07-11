import pandas as pd       
import matplotlib as mat
import matplotlib.pyplot as plt    
import numpy as np
import seaborn as sns

import pandas_profiling as pp
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

##################################################################################################################

path_train_data = 'Jobs/Kaggle/Competition - Insurance/input/train.csv'
path_test_data = 'Jobs/Kaggle/Competition - Insurance/input/test.csv'
path_submission = 'Jobs/Kaggle/Competition - Insurance/output/my_submission.csv'

df_train = pd.read_csv(path_train_data, index_col = 'id')
# print("\n", df_train)
# print("\n", df_train.info())
# print("\n", df_train.describe())

X_test = pd.read_csv(path_test_data, index_col = 'id')
# print("\n", X_test)
# print("\n", X_test.info())
# print("\n", X_test.describe())

# print("\n", pp.ProfileReport(df_train))

##################################################################################################################

X_train = df_train.copy().drop('target', axis = 1)
Y_train = df_train['target'].copy()

# List of categorical columns
object_cols = [col for col in X_train.columns if 'cat' in col]

# ordinal-encode categorical columns
ordinal_encoder = OrdinalEncoder()
X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
X_test[object_cols] = ordinal_encoder.transform(X_test[object_cols])

def cv_function(model, param, list):
    
    kfold = KFold(n_splits=5)
    search_model = model
    print ('Hyperparameter: ', param)
    
    for i in list:
        param_dict = {param : i}
        search_model.set_params(**param_dict)    
        cv_score = cross_val_score(search_model, X_train, Y_train, cv=kfold, scoring='neg_root_mean_squared_error')
        print("Parameter: {0:0.2f} - RMSE(SD): {1:0.4f} ({2:0.4f})". format(i, cv_score.mean(), cv_score.std()))


xgb_model = XGBRegressor(learning_rate = 0.02 ,random_state = 42, tree_method = 'gpu_hist')

params_xgb_list = [500,600,700,800,900,1000,1150,1300,1500]
param_xgb = 'n_estimators'
cv_function(xgb_model, param_xgb, params_xgb_list)

##################################################################################################################

xgb_final = XGBRegressor(n_estimators = 1500, learning_rate = 0.02 ,random_state = 42, tree_method = 'gpu_hist')

xgb_final.fit(X_train, Y_train)
predictions = xgb_final.predict(X_test)

# submission['target'] = predictions
# submission.to_csv(path_submission, index=False)

##################################################################################################################