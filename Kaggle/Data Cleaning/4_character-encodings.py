import pandas as pd

# helpful character encoding module
import chardet

path_police_killings = 'Jobs/Kaggle/Data Cleaning/input/PoliceKillingsUS.csv'
# police_killings = pd.read_csv(path_police_killings)

##################################################################################################################

""" CHARACTER ENCODINGS """

##################################################################################################################

""" STEP 1: WHAT ARE ENCODINGS? """

"""
You're working with a dataset composed of bytes. Run the code cell below to print a sample entry.
"""

sample_entry = b'\xa7A\xa6n'
print(sample_entry)
print('data type:', type(sample_entry))

"""
You notice that it doesn't use the standard UTF-8 encoding.
Use the next code cell to create a variable new_entry that changes the encoding from "big5-tw" to "utf-8". 
new_entry should have the bytes datatype.
"""

before = sample_entry.decode("big5-tw")
new_entry = before.encode()
print(new_entry)

##################################################################################################################

""" STEP 2: READING FILES IN ENCIDING PROBLEMS """

"""
Figure out what the correct encoding should be and read in the file to a DataFrame police_killings
"""

rawdata = open(path_police_killings, "rb").read()
result = chardet.detect(bytes(rawdata))
# print(result)
police_killings = pd.read_csv(path_police_killings, encoding=result['encoding'])

##################################################################################################################

""" STEP 3: SAVING FILES WITH UTF-8 ENCODING """

"""
Save a version of the police killings dataset to CSV with UTF-8 encoding. Your answer will be marked correct 
after saving this file.

Note: When using the to_csv() method, supply only the name of the file (e.g., "my_file.csv"). This saves the 
file at the filepath "/kaggle/working/my_file.csv"
"""

# UTF-8 by default
police_killings.to_csv("my_file.csv")

##################################################################################################################