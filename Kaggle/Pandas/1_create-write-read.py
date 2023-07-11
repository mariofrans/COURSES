import pandas as pd

# read file content
filename_150 = 'Jobs/Kaggle/Pandas/input/winemag-data-150k.csv'
reviews = pd.read_csv(filename_150, index_col=[0])

# save data to csv file
# reviews.to_csv(filename)

##################################################################################################################

""" CREATING, WRITING, READING """

# DataFrames
my_dictionary = {'Yes': [50, 21, 34, 4, 23], 'No': [131, 2, 50, 21, 6]}
index_list = ['2021', '2020', '2019', '2018', '2017']

print("\n===============================================================\n")

df_no_index = pd.DataFrame(my_dictionary)
print('DataFrame No Index:')
print(df_no_index)
print(" ")

df_index = pd.DataFrame(my_dictionary, index=index_list)
print('DataFrame With Index:')
print(df_index)

print("\n===============================================================\n")

# Series
contents_list = [30, 35, 40]
index_list = ['2015 Sales', '2016 Sales', '2017 Sales']

s_no_index = pd.Series(contents_list)
print('Series No Index:')
print(s_no_index)
print(" ")

s_index = pd.Series(contents_list, index=index_list, name='Toyota')
print('Series With Index:')
print(s_index)

print("\n===============================================================\n")

# create dataframe for multiple data
countries = ['Italy', 'Portugal', 'US', 'US', 'Indonesia', 'Japan']
provinces = ['Sicily & Sardinia', 'Douro', 'California', 'New York', 'Jakarta', 'Tokyo']
region_1 = ['Etna', None, 'Napa Valley', 'Finger Lakes', 'SCBD', 'Tokyo']
region_2 = [None, None, 'Napa', 'Finger Lakes', 'PIK', 'Shibuya']
points = [100, 90, 70, 99, 94, 98]

df_custom = pd.DataFrame(countries, columns=['country'])
df_custom['provinces'] = provinces
df_custom['region_1'] = region_1
df_custom['region_2'] = region_2
df_custom['points'] = points
print('Custom DataFrame:')
print(df_custom)

print("\n===============================================================\n")

##################################################################################################################


