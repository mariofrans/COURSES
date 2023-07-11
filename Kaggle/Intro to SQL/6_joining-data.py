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

""" JOINING DATA """

# Construct a reference to the "stackoverflow" dataset
dataset_ref = client.dataset("stackoverflow", project="bigquery-public-data")
# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

##################################################################################################################

""" STEP 1: EXPLORE THE DATA """

# Get a list of available tables 
list_of_tables = []
for table in list(client.list_tables(dataset)): list_of_tables.append(table.table_id)
print(list_of_tables)

##################################################################################################################

""" STEP 2: REVIEW RELEVANT TABLES """

# Construct a reference to the "posts_answers" table
answers_table_ref = dataset_ref.table("posts_answers")
# API request - fetch the table
answers_table = client.get_table(answers_table_ref)
# Preview the first five lines of the "posts_answers" table
client.list_rows(answers_table, max_results=5).to_dataframe()

# Construct a reference to the "posts_questions" table
questions_table_ref = dataset_ref.table("posts_questions")
# API request - fetch the table
questions_table = client.get_table(questions_table_ref)
# Preview the first five lines of the "posts_questions" table
client.list_rows(questions_table, max_results=5).to_dataframe()

##################################################################################################################

""" STEP 3: SELECT THE RIGHT QUESTIONS """

"""
Write a query that selects the id, title and owner_user_id columns from the posts_questions table
    1. Restrict the results to rows that contain the word "bigquery" in the tags column.
    2. Include rows where there is other text in addition to the word "bigquery" 
    (e.g., if a row has a tag "bigquery-sql", your results should include that too).
"""

questions_query = """
                  SELECT id, title, owner_user_id
                  FROM `bigquery-public-data.stackoverflow.posts_questions`
                  WHERE tags LIKE '%bigquery%'
                  """
questions_results = run_query(questions_query)
# print(questions_results.head())

##################################################################################################################

""" STEP 4: YOUR FIRST JOIN """

"""
Write a query that returns the id, body and owner_user_id columns from the posts_answers table for answers 
to "bigquery"-related questions.
    1. You should have one row in your results for each answer to a question that has "bigquery" in the tags.
    2. Remember you can get the tags for a question from the tags column in the posts_questions table.
"""

answers_query = """
                SELECT a.id, a.body, a.owner_user_id
                FROM `bigquery-public-data.stackoverflow.posts_questions` AS q 
                INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                ON q.id = a.parent_id
                WHERE q.tags LIKE '%bigquery%'
                """
answers_results = run_query(answers_query)
# print(answers_results.head())

##################################################################################################################

""" STEP 5: ANSWER THE QUESTION """

"""
You have the merge you need. But you want a list of users who have answered many questions... which requires 
more work beyond your previous result.

Write a new query that has a single row for each user who answered at least one question with a tag that includes 
the string "bigquery". Your results should have two columns:
    1. user_id - contains the owner_user_id column from the posts_answers table
    2. number_of_answers - contains the number of answers the user has written to "bigquery"-related questions
"""

bigquery_experts_query = """
                         SELECT a.owner_user_id AS user_id, COUNT(1) AS number_of_answers
                         
                         FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
                         INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                             ON q.id = a.parent_Id
                             
                         WHERE q.tags LIKE '%bigquery%'
                         GROUP BY a.owner_user_id
                         """
bigquery_experts_results = run_query(bigquery_experts_query)
# print(bigquery_experts_results.head())

##################################################################################################################
