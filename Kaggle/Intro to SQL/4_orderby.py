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

""" ORDER BY """

# Construct a reference to the "world_bank_intl_education" dataset
dataset_ref = client.dataset("world_bank_intl_education", project="bigquery-public-data")
# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "international_education" table
table_ref = dataset_ref.table("international_education")
# API request - fetch the table
table = client.get_table(table_ref)
# Preview the first five lines of the "international_education" table
client.list_rows(table, max_results=5).to_dataframe()

##################################################################################################################

""" STEP 1: GOVERNMENT EXPENDITURE ON EDUCATION """

"""
Which countries spend the largest fraction of GDP on education?
To answer this question, consider only the rows in the dataset corresponding to indicator code 
SE.XPD.TOTL.GD.ZS, and write a query that returns the average value in the value column for each 
country in the dataset between the years 2010-2017 (including 2010 and 2017 in the average).

Requirements:
    1. Your results should have the country name rather than the country code. You will have one row for 
    each country. 
    2. The aggregate function for average is AVG(). Use the name avg_ed_spending_pct for the column 
    created by this aggregation. 
    3. Order the results so the countries that spend the largest 
    fraction of GDP on education show up first.
"""

country_spend_pct_query = """
                          SELECT country_name, AVG(value) AS avg_ed_spending_pct

                          FROM `bigquery-public-data.world_bank_intl_education.international_education`

                          WHERE indicator_code='SE.XPD.TOTL.GD.ZS' and year >= 2010 and year <= 2017

                          GROUP BY country_name
                          ORDER BY avg_ed_spending_pct DESC
                          """
country_spending_results = run_query(country_spend_pct_query)
# print(country_spending_results.head())

##################################################################################################################

""" STEP 2: IDENTIFY INTERESTING CODES TO EXPLORE """

"""
Write a query below that selects the indicator code and indicator name for all codes with at least 
175 rows in the year 2016.

Requirements:
    1. You should have one row for each indicator code.
    2. The columns in your results should be called indicator_code, indicator_name, and num_rows.
    3. Only select codes with 175 or more rows in the raw database (exactly 175 rows would be included).
    4. To get both the indicator_code and indicator_name in your resulting DataFrame, you need to include 
    both in your SELECT statement (in addition to a COUNT() aggregation). This requires you to include 
    both in your GROUP BY clause.
    5. Order from results most frequent to least frequent.
"""

code_count_query = """
                    SELECT indicator_code, indicator_name, 
                        COUNT(1) AS num_rows

                    FROM `bigquery-public-data.world_bank_intl_education.international_education`

                    WHERE year=2016
                    HAVING COUNT(1)>=175

                    GROUP BY indicator_name, indicator_code
                    ORDER BY COUNT(1) DESC
                    """
code_count_results = run_query(code_count_query)
# print(code_count_results.head())

##################################################################################################################