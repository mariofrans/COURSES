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

""" GROUPBY, HAVING, & COUNT """

# Construct a reference to the "hacker_news" dataset
dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")
# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "comments" table
table_ref = dataset_ref.table("comments")
# API request - fetch the table
table = client.get_table(table_ref)
# Preview the first five lines of the "comments" table
client.list_rows(table, max_results=5).to_dataframe()

##################################################################################################################

""" STEP 1: PROLIFIC COMMENTERS """

"""
Write a query that returns all authors with more than 10,000 posts as well as their post counts. 
Call the column with post counts NumPosts.
"""

# Query to select prolific commenters and post counts
prolific_commenters_query = """
                            SELECT author, COUNT(1) AS NumPosts
                            FROM `bigquery-public-data.hacker_news.comments`
                            GROUP BY author
                            HAVING COUNT(1) > 10000
                            """
prolific_commenters = run_query(prolific_commenters_query)
# print(prolific_commenters.head())

##################################################################################################################

""" STEP 2: DELETED COMMENTS """

"""
How many comments have been deleted? 
(If a comment was deleted, the deleted column in the comments table will have the value True.)
"""

# Query to select all rows with attribute 'deleted=True'
deleted_post =  """
                SELECT *
                FROM `bigquery-public-data.hacker_news.comments`
                WHERE deleted=True
                """
deleted_post = run_query(deleted_post)
num_deleted_posts = len(deleted_post)
print(num_deleted_posts)

##################################################################################################################