from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

##################################################################################################################

""" GETTING STARTED WITH SQL & BIGQUERY """

# Construct a reference to the "chicago_crime" dataset
dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")
# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

##################################################################################################################

""" STEP 1: COUNT TABLES IN THE DATASET """

"""
How many tables are in the Chicago Crime dataset?
"""

# Store the answer as num_tables and then run this cell
num_tables = len(list(client.list_tables(dataset)))

##################################################################################################################

""" STEP 2: EXPLORE TABLE SCHEMA """

"""
How many columns in the crime table have TIMESTAMP data?
"""

# Construct a reference to the "crime" table
table_ref = dataset_ref.table("crime")
# API request - fetch the table
table = client.get_table(table_ref)
# Print information on all the columns in the "crime" table in the "chicago_crime" dataset
print(table.schema)
num_timestamp_fields = 2

##################################################################################################################

""" STEP 3: CREATE A CRIME MAP """

"""
If you wanted to create a map with a dot at the location of each crime, what are the names of the two fields 
you likely need to pull out of the crime table to plot the crimes on a map?
"""

df_table = client.list_rows(table, max_results=5).to_dataframe()
print(df_table)
fields_for_plotting = ['latitude', 'longitude']

##################################################################################################################