from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

##################################################################################################################

""" NESTED & REPEATED DATA """

# Construct a reference to the "github_repos" dataset
dataset_ref = client.dataset("github_repos", project="bigquery-public-data")
# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "sample_commits" table
table_ref = dataset_ref.table("sample_commits")
# API request - fetch the table
sample_commits_table = client.get_table(table_ref)
# Preview the first five lines of the table
client.list_rows(sample_commits_table, max_results=5).to_dataframe()

# Print information on all the columns in the table
sample_commits_table.schema

##################################################################################################################

""" STEP 1: WHO HAS THE MOST COMMITS IN 2016? """

"""
Write a query to find the individuals with the most commits in this table in 2016. Your query should return a 
table with two columns:
    1. committer_name - contains the name of each individual with a commit (from 2016) in the table
    2. num_commits - shows the number of commits the individual has in the table (from 2016)
Sort the table, so that people with more commits appear first.

Note: You can find the name of each committer and the date of the commit under the "committer" column, in 
the "name" and "date" child fields, respectively.
"""

max_commits_query = """
                    SELECT 
                    committer.name AS committer_name,
                    COUNT(*) AS num_commits
                    
                    FROM `bigquery-public-data.github_repos.sample_commits`
                    
                    WHERE committer.date >= '2016-01-01' AND committer.date < '2017-01-01'
                    GROUP BY committer_name
                    ORDER BY num_commits DESC
                    """

##################################################################################################################

""" STEP 2: LOOK AT LANGUAGES """

# Construct a reference to the "languages" table
table_ref = dataset_ref.table("languages")
# API request - fetch the table
languages_table = client.get_table(table_ref)
# Preview the first five lines of the table
client.list_rows(languages_table, max_results=5).to_dataframe()

# Print information on all the columns in the table
languages_table.schema

"""
Each row of the languages table corresponds to a different repository.
    1. The "repo_name" column contains the name of the repository,
    2. the "name" field in the "language" column contains the programming languages that can be found in the repo, and
    3. the "bytes" field in the "language" column has the size of the files (in bytes, for the corresponding language).
"""

##################################################################################################################

""" STEP 3: WHAT IS THE MOST POPULAR PROGRAMMING LANGUAGE? """

"""
Write a query to leverage the information in the languages table to determine which programming languages appear 
in the most repositories. The table returned by your query should have two columns:
    1. language_name - the name of the programming language
    2. num_repos - the number of repositories in the languages table that use the programming language
Sort the table so that languages that appear in more repos are shown first.
"""

pop_lang_query = """
                 SELECT l.name as language_name, COUNT(*) as num_repos

                 FROM `bigquery-public-data.github_repos.languages`,
                     UNNEST(language) AS l

                 GROUP BY language_name
                 ORDER BY num_repos DESC
                 """

##################################################################################################################

""" STEP 4: WHICH LANGUAGES ARE USED MOST IN THE REPOSITORY? """

"""
For this question, you'll restrict your attention to the repository with name 'polyrabbit/polyglot'.

Write a query that returns a table with one row for each language in this repository. The table should have two 
columns:
1. name - the name of the programming language
2. bytes - the total number of bytes of that programming language
Sort the table by the bytes column so that programming languages that take up more space in the repo appear first.
"""

all_langs_query = """
                    SELECT l.name, l.bytes
                        
                    FROM `bigquery-public-data.github_repos.languages`,
                        UNNEST(language) AS l
                    
                    WHERE repo_name = 'polyrabbit/polyglot'
                    ORDER BY l.bytes DESC
                  """

##################################################################################################################