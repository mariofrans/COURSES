import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.plotting.register_matplotlib_converters()

path_cancer_b = "Jobs/Kaggle/Data Visualization/input/cancer_b.csv"
path_cancer_m = "Jobs/Kaggle/Data Visualization/input/cancer_m.csv"

##################################################################################################################

""" DISTRIBUTIONS """

##################################################################################################################

""" STEP 1: LOAD THE DATA """

cancer_b_data = pd.read_csv(path_cancer_b, index_col='Id')
cancer_m_data = pd.read_csv(path_cancer_m, index_col='Id')

##################################################################################################################

""" STEP 2: REVIEW THE DATA """

# Print the first five rows of the (benign) data
print(cancer_b_data.head())
# Print the first five rows of the (malignant) data
print(cancer_m_data.head())

# FIn the first five rows of the data for benign tumors, what is the largest value for 'Perimeter (mean)'?
max_perim = cancer_b_data.head()['Perimeter (mean)'].max()
# What is the value for 'Radius (mean)' for the tumor with Id 842517?
mean_radius = cancer_m_data.head().loc[842517]['Radius (mean)']

##################################################################################################################

""" STEP 3: INVESTIGATING DIFFERENCES """

# Histograms for benign and maligant tumors
sns.distplot(cancer_b_data['Area (mean)'], label='Benign', kde=False)
sns.distplot(cancer_m_data['Area (mean)'], label='Maligant', kde=False)
plt.legend()

"""
A researcher approaches you for help with identifying how the 'Area (mean)' column can be used to understand 
the difference between benign and malignant tumors. Based on the histograms above,
    1. Do malignant tumors have higher or lower values for 'Area (mean)' (relative to benign tumors), on average?
    2. Which tumor type seems to have a larger range of potential values?
"""

"""
Malignant tumors have higher values for 'Area (mean)', on average. Malignant tumors have a larger range of 
potential values
"""

##################################################################################################################

""" STEP 4: A VERY USEFUL COLUMN """

sns.kdeplot(cancer_b_data['Radius (worst)'], label='Benign',shade=True)
sns.kdeplot(cancer_m_data['Radius (worst)'], label='Maligant', shade=True)
plt.legend()

"""
A hospital has recently started using an algorithm that can diagnose tumors with high accuracy. Given a tumor 
with a value for 'Radius (worst)' of 25, do you think the algorithm is more likely to classify the tumor as 
benign or malignant?
"""

"""
The algorithm is more likely to classify the tumor as malignant. This is because the curve for malignant tumors 
is much higher than the curve for benign tumors around a value of 25 -- and an algorithm that gets high accuracy 
is likely to make decisions based on this pattern in the data.
"""

##################################################################################################################