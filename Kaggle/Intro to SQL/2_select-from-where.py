from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

##################################################################################################################

def run_query(query):
    # Set up the query
    safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
    results = client.query(query, job_config=safe_config)
    # API request - run the query, and return a pandas DataFrame
    df_results = results.to_dataframe()
    return df_results

##################################################################################################################

""" SELECT, FROM, & WHERE """

# Construct a reference to the "openaq" dataset
dataset_ref = client.dataset("openaq", project="bigquery-public-data")
# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "global_air_quality" table
table_ref = dataset_ref.table("global_air_quality")
# API request - fetch the table
table = client.get_table(table_ref)
# Preview the first five lines of the "global_air_quality" table
client.list_rows(table, max_results=5).to_dataframe()

##################################################################################################################

""" STEP 1: UNITS OF MEASUREMENT """

"""
Which countries have reported pollution levels in units of "ppm"? In the code cell below, set first_query 
to an SQL query that pulls the appropriate entries from the country column.
"""

# Query to select countries with units of "ppm"
first_query =   """
                SELECT country 
                FROM `bigquery-public-data.openaq.global_air_quality` 
                WHERE unit = "ppm" 
                """
first_results = run_query(first_query)
# print(first_results.head())

##################################################################################################################

""" STEP 2: HIGH AIR QUALITY """

"""
Which pollution levels were reported to be exactly 0?
    1. Set zero_pollution_query to select all columns of the rows where the value column is 0.
    2. Set zero_pollution_results to a pandas DataFrame containing the query results.
"""

# Query to select all columns where pollution levels are exactly 0
zero_pollution_query =  """
                        SELECT * 
                        FROM `bigquery-public-data.openaq.global_air_quality` 
                        WHERE value=0
                        """
zero_pollution_results = run_query(zero_pollution_query)
# print(zero_pollution_results.head())

##################################################################################################################