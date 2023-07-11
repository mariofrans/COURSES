import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

# Path of the file to be read
path_train_data = 'Jobs/Kaggle/Intermediate Machine Learning/input/train.csv'
path_test_data = 'Jobs/Kaggle/Intermediate Machine Learning/input/test.csv'
path_submission = 'Jobs/Kaggle/Intermediate Machine Learning/output/submission.csv'

# Read the data
train_data = pd.read_csv(path_train_data, index_col='Id')
test_data = pd.read_csv(path_test_data, index_col='Id')

##################################################################################################################

# Remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice              
train_data.drop(['SalePrice'], axis=1, inplace=True)

# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()
X.head()

##################################################################################################################

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')

print("Average MAE score:", scores.mean())

##################################################################################################################

""" STEP 1: WRITE A USEFUL FUNCTION """

def get_score(n_estimators):

    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators, random_state=0))
    ])
    scores = -1 * cross_val_score(my_pipeline, X, y, cv=3, scoring='neg_mean_absolute_error')
    return scores.mean()

##################################################################################################################

""" STEP 2: TEST DIFFERENT PARAMETER VALUES """

results = {}
for n_estimators in [50, 100, 150, 200, 250, 300, 350, 400]: 
    results[n_estimators] = get_score(n_estimators)

plt.plot(list(results.keys()), list(results.values()))
plt.show()

##################################################################################################################

""" STEP 3: FIND THE BEST PARAMETER VALUE """

results_values = list(results.values())
results_keys = list(results.keys())

smallest_result_value_index = results_values.index(min(results_values)) 
n_estimators_best = results_keys[smallest_result_value_index]
# n_estimators_best = min(results, key=results.get)

##################################################################################################################