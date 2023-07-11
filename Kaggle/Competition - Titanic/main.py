import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Path of the file to be read
path_gender_submission = 'Jobs/Kaggle/Titanic/input/gender_submission.csv'
path_train_data = 'Jobs/Kaggle/Titanic/input/train.csv'
path_test_data = 'Jobs/Kaggle/Titanic/input/test.csv'
path_submission = 'Jobs/Kaggle/Titanic/output/my_submission.csv'

gender_submission = pd.read_csv(path_gender_submission)
train_data = pd.read_csv(path_train_data)
test_data = pd.read_csv(path_test_data)

##################################################################################################################

# find the percentage of women who survived
women_all = train_data.loc[ train_data['Sex']=='female' ]
women_survived = women_all.loc[ women_all['Survived']==1 ]
rate_women = len(women_survived)/len(women_all)
# print("Percentage of Women Who Survived:", rate_women)
# print(women_all)
# print(women_survived)

# find the percentage of men who survived
men_all = train_data.loc[ train_data['Sex']=='male' ]
men_survived = men_all.loc[ men_all['Survived']==1 ]
rate_men = len(men_survived)/len(men_all)
# print("Percentage of Men Who Survived:", rate_men)
# print(men_all)
# print(men_survived)

##################################################################################################################

# the results column
y = train_data["Survived"]
# columns which 'potentially' affects the results'
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# machine learning to find patterns
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

# save output to csv file
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv(path_submission, index=False)
print("Your submission was successfully saved!")

##################################################################################################################