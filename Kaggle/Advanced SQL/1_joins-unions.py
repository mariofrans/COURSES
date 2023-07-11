from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

##################################################################################################################

""" JOINs & UNIONs """

# Construct a reference to the "stackoverflow" dataset
dataset_ref = client.dataset("stackoverflow", project="bigquery-public-data")
# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "posts_questions" table
table_ref = dataset_ref.table("posts_questions")
# API request - fetch the table
table = client.get_table(table_ref)
# Preview the first five lines of the table
client.list_rows(table, max_results=5).to_dataframe()

# Construct a reference to the "posts_answers" table
table_ref = dataset_ref.table("posts_answers")
# API request - fetch the table
table = client.get_table(table_ref)
# Preview the first five lines of the table
client.list_rows(table, max_results=5).to_dataframe()

##################################################################################################################

""" STEP 1: HOW LONG DOES IT TAKE FOR QUESTIONS TO RECEIVE ANSWERS? """

"""
You're interested in exploring the data to have a better understanding of how long it generally takes for 
questions to receive answers. Armed with this knowledge, you plan to use this information to better design 
the order in which questions are presented to Stack Overflow users.

With this goal in mind, you write the query below, which focuses on questions asked in January 2018. It 
returns a table with two columns:
    1. q_id - the ID of the question
    2. time_to_answer - how long it took (in seconds) for the question to receive an answer
"""

first_query = """
              SELECT q.id AS q_id,
                  MIN(TIMESTAMP_DIFF(a.creation_date, q.creation_date, SECOND)) as time_to_answer

              FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
                  INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                    ON q.id = a.parent_id

              WHERE q.creation_date >= '2018-01-01' and q.creation_date < '2018-02-01'

              GROUP BY q_id
              ORDER BY time_to_answer
              """

first_result = client.query(first_query).result().to_dataframe()
print("Percentage of answered questions: %s%%" % \
      (sum(first_result["time_to_answer"].notnull()) / len(first_result) * 100))
print("Number of questions:", len(first_result))
first_result.head()

"""
You're surprised at the results and strongly suspect that something is wrong with your query. In particular,
    1. According to the query, 100% of the questions from January 2018 received an answer. But, you know that 
    ~80% of the questions on the site usually receive an answer.
    2. The total number of questions is surprisingly low. You expected to see at least 150,000 questions 
    represented in the table.

Given these observations, you think that the type of JOIN you have chosen has inadvertently excluded unanswered 
questions. Using the code cell below, can you figure out what type of JOIN to use to fix the problem so that the 
table includes unanswered questions?

Note: You need only amend the type of JOIN (i.e., INNER, LEFT, RIGHT, or FULL) to answer the question successfully.
"""

correct_query = """
                SELECT q.id AS q_id,
                    MIN(TIMESTAMP_DIFF(a.creation_date, q.creation_date, SECOND)) as time_to_answer

                FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
                    FULL JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                        ON q.id = a.parent_id

                WHERE q.creation_date >= '2018-01-01' and q.creation_date < '2018-02-01'

                GROUP BY q_id
                ORDER BY time_to_answer
                """

first_result = client.query(first_query).result().to_dataframe()
print("Percentage of answered questions: %s%%" % \
      (sum(first_result["time_to_answer"].notnull()) / len(first_result) * 100))
print("Number of questions:", len(first_result))
first_result.head()

##################################################################################################################

""" STEP 2: INITIAL QUESTIONS & ANSWERS, PART 1 """

"""
You're interested in understanding the initial experiences that users typically have with the Stack Overflow 
website. Is it more common for users to first ask questions or provide answers? After signing up, how long 
does it take for users to first interact with the website? To explore this further, you draft the (partial) 
query in the code cell below.

The query returns a table with three columns:
1. owner_user_id - the user ID
2. q_creation_date - the first time the user asked a question
3. a_creation_date - the first time the user contributed an answer

You want to keep track of users who have asked questions, but have yet to provide answers. And, your table should 
also include users who have answered questions, but have yet to pose their own questions.

With this in mind, please fill in the appropriate JOIN (i.e., INNER, LEFT, RIGHT, or FULL) to return the correct 
information.

Note: You need only fill in the appropriate JOIN. All other parts of the query should be left as-is. (You also don't 
need to write any additional code to run the query, since the cbeck() method will take care of this for you.)

To avoid returning too much data, we'll restrict our attention to questions and answers posed in January 2019. We'll 
amend the timeframe in Part 2 of this question to be more realistic!
"""

q_and_a_query = """
                SELECT q.owner_user_id AS owner_user_id,
                    MIN(q.creation_date) AS q_creation_date,
                    MIN(a.creation_date) AS a_creation_date

                FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
                    FULL JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                        ON q.owner_user_id = a.owner_user_id 

                WHERE q.creation_date >= '2019-01-01' AND q.creation_date < '2019-02-01' 
                    AND a.creation_date >= '2019-01-01' AND a.creation_date < '2019-02-01'

                GROUP BY owner_user_id
                """

##################################################################################################################

""" STEP 3: INITIAL QUESTIONS & ANSWERS, PART 2 """

"""
To answer this question, you'll need to pull information from three different tables! This syntax very similar 
to the case when we have to join only two tables. For instance, consider the three tables below.

We can use two different JOINs to link together information from all three tables, in a single query.

With this in mind, say you're interested in understanding users who joined the site in January 2019. You want to 
track their activity on the site: when did they post their first questions and answers, if ever?

Write a query that returns the following columns:

1. id - the IDs of all users who created Stack Overflow accounts in January 2019 (January 1, 2019, to January 31, 
2019, inclusive)
2. q_creation_date - the first time the user posted a question on the site; if the user has never posted a question, 
the value should be null
3. a_creation_date - the first time the user posted a question on the site; if the user has never posted a question, 
the value should be null

Note that questions and answers posted after January 31, 2019, should still be included in the results. And, all 
users who joined the site in January 2019 should be included (even if they have never posted a question or 
provided an answer).

The query from the previous question should be a nice starting point to answering this question! You'll need to 
use the posts_answers and posts_questions tables. You'll also need to use the users table from the Stack Overflow 
dataset. The relevant columns from the users table are id (the ID of each user) and creation_date (when the user 
joined the Stack Overflow site, in DATETIME format).
"""

# Construct a reference to the "users" table
table_ref = dataset_ref.table("users")
# API request - fetch the table
table = client.get_table(table_ref)
# Preview the first five lines of the table
client.list_rows(table, max_results=5).to_dataframe()

three_tables_query = """
                    SELECT u.id AS id,
                        MIN(q.creation_date) AS q_creation_date,
                        MIN(a.creation_date) AS a_creation_date

                    FROM `bigquery-public-data.stackoverflow.users` AS u
                        LEFT JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                            ON u.id = a.owner_user_id
                        LEFT JOIN `bigquery-public-data.stackoverflow.posts_questions` AS q
                            ON q.owner_user_id = u.id

                    WHERE u.creation_date >= '2019-01-01' and u.creation_date < '2019-02-01'

                    GROUP BY id
                    """

##################################################################################################################

""" STEP 4: HOW MANY DISTINCT USERS POSTED ON JANUARY 1, 2019? """

"""
In the code cell below, write a query that returns a table with a single column:
    1. owner_user_id - the IDs of all users who posted at least one question or answer on January 1, 2019. Each 
    user ID should appear at most once.

In the posts_questions (and posts_answers) tables, you can get the ID of the original poster from the 
owner_user_id column. Likewise, the date of the original posting can be found in the creation_date column.
"""

all_users_query = """
                  SELECT q.owner_user_id 
                  
                  FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
                  WHERE EXTRACT(DATE FROM q.creation_date) = '2019-01-01'
                  UNION DISTINCT
                  
                  SELECT a.owner_user_id
                  FROM `bigquery-public-data.stackoverflow.posts_answers` AS a
                  WHERE EXTRACT(DATE FROM a.creation_date) = '2019-01-01'
                  """

##################################################################################################################