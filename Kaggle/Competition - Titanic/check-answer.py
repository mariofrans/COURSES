import pandas as pd

path_answer = 'Jobs/Kaggle/Competition - Titanic/output/my_submission_1.00000_A.csv'
path_submission = 'Jobs/Kaggle/Competition - Titanic/output/my_submission.csv'

df_answer = pd.read_csv(path_answer)
df_submission = pd.read_csv(path_submission)

print(len(df_answer))
print(len(df_submission))

count = 0
for i in range(len(df_submission)):
    if df_submission['Survived'].iloc[i]==df_answer['Survived'].iloc[i]: count += 1

print("Score:", count/len(df_answer))