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

""" AS & WITH """

# Construct a reference to the "chicago_taxi_trips" dataset
dataset_ref = client.dataset("chicago_taxi_trips", project="bigquery-public-data")
# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

##################################################################################################################

""" STEP 1-2: FIND & PEEK THE DATA """

# List all the tables in the dataset
tables = list(client.list_tables(dataset))
for table in tables: print(table.table_id)
# Write the table name as a string below
table_name = 'taxi_trips'

# Construct a reference to the "taxi_trips" table
table_ref = dataset_ref.table("taxi_trips")
# API request - fetch the table
table = client.get_table(table_ref)
# Preview the first five lines of the "taxi_trips" table
client.list_rows(table, max_results=5).to_dataframe()

##################################################################################################################

""" STEP 3: DETERMINE WHERE THE DATA IS FROM """

"""
If the data is sufficiently old, we might be careful before assuming the data is still relevant to traffic 
patterns today. Write a query that counts the number of trips in each year.
Your results should have two columns:
    1. year - the year of the trips
    2. num_trips - the number of trips in that year
Hints:
    1. When using GROUP BY and ORDER BY, you should refer to the columns by the alias year that you set at the top 
    of the SELECT query.
    2. The SQL code to SELECT the year from trip_start_timestamp is SELECT EXTRACT(YEAR FROM trip_start_timestamp)
    3. The FROM field can be a little tricky until you are used to it. The format is:
        A backick (the symbol `).
        The project name. In this case it is bigquery-public-data.
        A period.
        The dataset name. In this case, it is chicago_taxi_trips.
        A period.
        The table name. You used this as your answer in 1) Find the data.
        A backtick (the symbol `).
"""

rides_per_year_query = """
                       SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, 
                              COUNT(1) AS num_trips

                       FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                       GROUP BY year
                       ORDER BY year
                       """
rides_per_year_result = run_query(rides_per_year_query)
# print(rides_per_year_result)

##################################################################################################################

""" STEP 4: DIVE SLIGHTLY DEEPER """

"""
You'd like to take a closer look at rides from 2017. Copy the query you used above in rides_per_year_query into 
the cell below for rides_per_month_query. Then modify it in two ways:
    1. Use a WHERE clause to limit the query to data from 2017.
    2. Modify the query to extract the month rather than the year.
"""

rides_per_month_query = """
                       SELECT EXTRACT(MONTH FROM trip_start_timestamp) AS month, 
                              COUNT(1) AS num_trips

                       FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                       WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2017

                       GROUP BY month
                       ORDER BY month
                       """
rides_per_month_result = run_query(rides_per_month_query)
print(rides_per_month_result)

##################################################################################################################

""" STEP 5: WRITE THE QUERY """

"""
It's time to step up the sophistication of your queries. Write a query that shows, for each hour of the day in 
the dataset, the corresponding number of trips and average speed.

Your results should have three columns:
1. hour_of_day - sort by this column, which holds the result of extracting the hour from trip_start_timestamp.
2. num_trips - the count of the total number of trips in each hour of the day (e.g. how many trips were started 
between 6AM and 7AM, independent of which day it occurred on).
3. avg_mph - the average speed, measured in miles per hour, for trips that started in that hour of the day. 
Average speed in miles per hour is calculated as 3600 * SUM(trip_miles) / SUM(trip_seconds). (The value 3600 is 
used to convert from seconds to hours.)

Restrict your query to data meeting the following criteria:
1. a trip_start_timestamp between 2017-01-01 and 2017-07-01
2. trip_seconds > 0 and trip_miles > 0

You will use a common table expression (CTE) to select just the relevant rides. Because this dataset is very big, 
this CTE should select only the columns you'll need to create the final output (though you won't actually create 
those in the CTE -- instead you'll create those in the later SELECT statement below the CTE).
"""

speeds_query = """
               WITH RelevantRides AS
               (
                   SELECT EXTRACT(HOUR FROM trip_start_timestamp) AS hour_of_day, 
                          trip_miles, 
                          trip_seconds

                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                   WHERE trip_start_timestamp > '2017-01-01' AND 
                         trip_start_timestamp < '2017-07-01' AND 
                         trip_seconds > 0 AND 
                         trip_miles > 0
               )
               
               SELECT hour_of_day, 
                      COUNT(1) AS num_trips, 
                      3600 * SUM(trip_miles) / SUM(trip_seconds) AS avg_mph

               FROM RelevantRides

               GROUP BY hour_of_day
               ORDER BY hour_of_day
               """
speeds_result = run_query(speeds_query)
print(speeds_result)

##################################################################################################################
