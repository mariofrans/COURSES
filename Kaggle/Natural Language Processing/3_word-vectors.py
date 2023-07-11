import pandas as pd
import numpy as np
import spacy
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

path_data = 'Jobs/Kaggle/Natural Language Processing/input/yelp_ratings.csv'
path_vectors = 'Jobs/Kaggle/Natural Language Processing/input/review_vectors.npy'

##################################################################################################################

""" WORD VECTORS """

"""
Embeddings are both conceptually clever and practically effective.

So let's try them for the sentiment analysis model you built for the restaurant. Then you can find the most 
similar review in the data set given some example text. It's a task where you can easily judge for yourself 
how well the embeddings work.
"""

nlp = spacy.load('en_core_web_lg')

review_data = pd.read_csv(path_data)
# print(review_data.head())

reviews = review_data[:100]

# We just want the vectors so we can turn off other models in the pipeline
with nlp.disable_pipes(): vectors = np.array([nlp(review.text).vector for idx, review in reviews.iterrows()])
# print(vectors.shape)

"""
The result is a matrix of 100 rows and 300 columns.

Why 100 rows? Because we have 1 row for each column.

Why 300 columns? This is the same length as word vectors. See if you can figure out why document vectors have 
the same length as word vectors (some knowledge of linear algebra or vector math would be needed to figure 
this out).
"""

# Loading all document vectors from file
vectors = np.load(path_vectors)

##################################################################################################################

""" STEP 1: TRAINING MODEL ON DOCUMENT VECTORS """

"""
Next you'll train a LinearSVC model using the document vectors. It runs pretty quick and works well in high 
dimensional settings like you have here.

After running the LinearSVC model, you might try experimenting with other types of models to see whether it 
improves your results.
"""

X_train, X_test, y_train, y_test = train_test_split(vectors, review_data.sentiment, test_size=0.1, random_state=1)

# Create the LinearSVC model
model = LinearSVC(random_state=1, dual=False)
# Fit the model
model.fit(X_train, y_train)

# Uncomment and run to see model accuracy
print(f'Model test accuracy: {model.score(X_test, y_test)*100:.3f}%')
# Output: Model test accuracy: 93.847%

##################################################################################################################

""" STEP 2: CENTERING THE VECTORS """

"""
Sometimes people center document vectors when calculating similarities. That is, they calculate the mean vector 
from all documents, and they subtract this from each individual document's vector. Why do you think this could 
help with similarity metrics?
"""

"""
Sometimes your documents will already be fairly similar. For example, this data set is all reviews of businesses. 
There will be stong similarities between the documents compared to news articles, technical manuals, and recipes. 
You end up with all the similarities between 0.8 and 1 and no anti-similar documents (similarity < 0). When the 
vectors are centered, you are comparing documents within your dataset as opposed to all possible documents.
"""

##################################################################################################################

""" STEP 3: FIND THE MOST SIMILAR REVIEW """

"""
Given an example review below, find the most similar document within the Yelp dataset using the cosine similarity.
"""

review = """
            I absolutely love this place. The 360 degree glass windows with the 
            Yerba buena garden view, tea pots all around and the smell of fresh tea everywhere 
            transports you to what feels like a different zen zone within the city. I know 
            the price is slightly more compared to the normal American size, however the food 
            is very wholesome, the tea selection is incredible and I know service can be hit 
            or miss often but it was on point during our most recent visit. Definitely recommend!

            I would especially recommend the butternut squash gyoza.
         """

def cosine_similarity(a, b): return np.dot(a, b)/np.sqrt(a.dot(a)*b.dot(b))
review_vec = nlp(review).vector

## Center the document vectors
# Calculate the mean for the document vectors
vec_mean = vectors.mean(axis=0)
# Subtract the mean from the vectors
centered = vectors - vec_mean

# Calculate similarities for each document in the dataset
# Make sure to subtract the mean from the review vector
sims = np.array([cosine_similarity(review_vec - vec_mean, vec) for vec in centered])
# Get the index for the most similar document
most_similar = sims.argmax()

print(review_data.iloc[most_similar].text)

# Output:

# After purchasing my final christmas gifts at the Urban Tea Merchant in Vancouver, I was surprised to hear 
# about Teopia at the new outdoor mall at Don Mills and Lawrence when I went back home to Toronto for Christmas.
# Across from the outdoor skating rink and perfect to sit by the ledge to people watch, the location was prime 
# for tea connesieurs... or people who are just freezing cold in need of a drinK!

# Like any gourmet tea shop, there were large tins of tea leaves on the walls, and although the tea menu seemed 
# interesting enough, you can get any specialty tea as your drink. We didn't know what to get... so the lady 
# suggested the Goji Berries... it smelled so succulent and juicy... instantly SOLD! I got it into a tea latte 
# and watched the tea steep while the milk was steamed, and surprisingly, with the click of a button, all the 
# water from the tea can be instantly drained into the cup (see photo).. very fascinating!

# The tea was aromatic and tasty, not over powering. The price was also very reasonable and I recommend everyone 
# to get a taste of this place :)

##################################################################################################################

""" STEP 4: LOOKING AT SIMILAR REVIEWS """

"""
If you look at other similar reviews, you'll see many coffee shops. Why do you think reviews for coffee are 
similar to the example review which mentions only tea?
"""

"""
Reviews for coffee shops will also be similar to our tea house review because coffee and tea are semantically 
similar. Most cafes serve both coffee and tea so you'll see the terms appearing together often.
"""

##################################################################################################################