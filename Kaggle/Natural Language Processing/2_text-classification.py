import pandas as pd
import random
import spacy
from spacy.matcher import PhraseMatcher
from spacy.util import minibatch
from collections import defaultdict

path_data = 'Jobs/Kaggle/Natural Language Processing/input/yelp_ratings.csv'

##################################################################################################################

""" TEXT CLASSIFICATION """

"""
You did such a great job for DeFalco's restaurant in the previous exercise that the chef has hired you for a 
new project.

The restaurant's menu includes an email address where visitors can give feedback about their food.

The manager wants you to create a tool that automatically sends him all the negative reviews so he can fix them, 
while automatically sending all the positive reviews to the owner, so the manager can ask for a raise.

You will first build a model to distinguish positive reviews from negative reviews using Yelp reviews because 
these reviews include a rating with each review. Your data consists of the text body of each review along with 
the star rating. Ratings with 1-2 stars count as "negative", and ratings with 4-5 stars are "positive". Ratings 
with 3 stars are "neutral" and have been dropped from the data.
"""

##################################################################################################################

""" STEP 1: EVALUATE THE APPROACH """

"""
Is there anything about this approach that concerns you?
"""

"""
Any way of setting up an ML problem will have multiple strengths and weaknesses. So you may have thought of 
different issues than listed here.

The strength of this approach is that it allows you to distinguish positive email messages from negative 
emails even though you don't have historical emails that you have labeled as positive or negative.

The weakness of this approach is that emails may be systematically different from Yelp reviews in ways that 
make your model less accurate. For example, customers might generally use different words or slang in emails, 
and the model based on Yelp reviews won't have seen these words.

If you wanted to see how serious this issue is, you could compare word frequencies between the two sources. 
In practice, manually reading a few emails from each source may be enough to see if it's a serious issue.

If you wanted to do something fancier, you could create a dataset that contains both Yelp reviews and emails 
and see whether a model can tell a reviews source from the text content. Ideally, you'd like to find that 
model didn't perform well, because it would mean your data sources are similar. That approach seems 
unnecessarily complex here.
"""

##################################################################################################################

""" STEP 2: REVIEW DATA & CREATE MODEL """

"""
Moving forward with your plan, you'll need to load the data. Here's some basic code to load data and split 
it into a training and validation set. Run this code
"""

def load_data(csv_file, split=0.9):
    data = pd.read_csv(csv_file)
    # Shuffle data
    train_data = data.sample(frac=1, random_state=7)
    texts = train_data.text.values
    labels = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in train_data.sentiment.values]
    split = int(len(train_data) * split)
    train_labels = [{"cats": labels} for labels in labels[:split]]
    val_labels = [{"cats": labels} for labels in labels[split:]]
    return texts[:split], train_labels, texts[split:], val_labels

train_texts, train_labels, val_texts, val_labels = load_data(path_data)

"""
You will use this training data to build a model. The code to build the model is the same as what you saw in 
the tutorial. So that is copied below for you.

But because your data is different, there are two lines in the modeling code cell that you'll need to change. 
Can you figure out what they are?

First, run the cell below to look at a couple elements from your training data
"""

print('Texts from training data\n------')
print(train_texts[:2])
print('\nLabels from training data\n------')
print(train_labels[:2])

# Create an empty model
nlp = spacy.blank("en")

# Create the TextCategorizer with exclusive classes and "bow" architecture
textcat = nlp.create_pipe( "textcat", config={  "exclusive_classes": True, "architecture": "bow"} )

# Add the TextCategorizer to the empty model
nlp.add_pipe(textcat)

# Add labels to text classifier
textcat.add_label("NEGATIVE")
textcat.add_label("POSITIVE")

##################################################################################################################

""" STEP 3: TRAIN FUNCTION """

"""
Implement a function train that updates a model with training data. Most of this is general data munging, 
which we've filled in for you. Just add the one line of code necessary to update your model.
"""

def train(model, train_data, optimizer):
    losses = {}
    random.seed(1)
    random.shuffle(train_data)
    
    batches = minibatch(train_data, size=8)
    for batch in batches:
        # train_data is a list of tuples [(text0, label0), (text1, label1), ...]
        # Split batch into texts and labels
        texts, labels = zip(*batch)
        # Update model with texts and labels
        model.update(texts, labels, sgd=optimizer, losses=losses)

    return losses

# Fix seed for reproducibility
spacy.util.fix_random_seed(1)
random.seed(1)

# This may take a while to run!
optimizer = nlp.begin_training()
train_data = list(zip(train_texts, train_labels))
losses = train(nlp, train_data, optimizer)
print(losses['textcat'])
# Output: 8.701575470772399

"""
We can try this slightly trained model on some example text and look at the probabilities assigned to each label.
"""

text = "This tea cup was full of holes. Do not recommend."
doc = nlp(text)
print(doc.cats)

"""
These probabilities look reasonable. Now you should turn them into an actual prediction.
"""

##################################################################################################################

""" STEP 4: MAKING PREDICTIONS """

"""
Implement a function predict that predicts the sentiment of text examples.
    1. First, tokenize the texts using nlp.tokenizer().
    2. Then, pass those docs to the TextCategorizer which you can get from nlp.get_pipe().
    3. Use the textcat.predict() method to get scores for each document, then choose the class with the 
    highest score (probability) as the predicted class.
"""

def predict(nlp, texts):
    # Use the tokenizer to tokenize each input text example
    docs = [nlp.tokenizer(text) for text in texts]

    # Use textcat to get the scores for each doc
    textcat = nlp.get_pipe('textcat')
    scores, _ = textcat.predict(docs)

    # From the scores, find the class with the highest score/probability
    predicted_class = scores.argmax(axis=1)
    return predicted_class

texts = val_texts[34:38]
predictions = predict(nlp, texts)

for p, t in zip(predictions, texts): print(f"{textcat.labels[p]}: {t} \n")

"""
It looks like your model is working well after going through the data just once. However you need to calculate 
some metric for the model's performance on the hold-out validation data.
"""

##################################################################################################################

""" STEP 5: EVALUATE THE MODEL """

"""
Implement a function that evaluates a TextCategorizer model. This function evaluate takes a model along with 
texts and labels. It returns the accuracy of the model, which is the number of correct predictions divided by 
all predictions.

First, use the predict method you wrote earlier to get the predicted class for each text in texts. Then, find 
where the predicted labels match the true "gold-standard" labels and calculate the accuracy.
"""

def evaluate(model, texts, labels):
    """ 
    Returns the accuracy of a TextCategorizer model. 
    Arguments
    ---------
    model: ScaPy model with a TextCategorizer
    texts: Text samples, from load_data function
    labels: True labels, from load_data function
    """
    
    # Get predictions from textcat model (using your predict method)
    predicted_class = predict(model, texts)
    
    # From labels, get the true class as a list of integers (POSITIVE -> 1, NEGATIVE -> 0)
    true_class = [int(each['cats']['POSITIVE']) for each in labels]
    
    # A boolean or int array indicating correct predictions
    correct_predictions = predicted_class == true_class
    
    # The accuracy, number of correct predictions divided by all predictions
    accuracy = correct_predictions.mean()
    return accuracy

accuracy = evaluate(nlp, val_texts, val_labels)
print(f"Accuracy: {accuracy:.4f}")
# Output: Accuracy: 0.9486

"""
With the functions implemented, you can train and evaluate in a loop.
"""

n_iters = 5
for i in range(n_iters):
    losses = train(nlp, train_data, optimizer)
    accuracy = evaluate(nlp, val_texts, val_labels)
    print(f"Loss: {losses['textcat']:.3f} \t Accuracy: {accuracy:.3f}")

##################################################################################################################

""" STEP 6: KEEP IMPROVING """

"""
You've built the necessary components to train a text classifier with spaCy. What could you do further to 
optimize the model?
"""

"""
Answer: There are various hyperparameters to work with here. The biggest one is the TextCategorizer architecture. 
You used the simplest model which trains faster but likely has worse performance than the CNN and ensemble models.
"""

##################################################################################################################