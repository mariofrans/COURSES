import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

path_red_wine = 'Jobs/Kaggle/Intro to Deep Learning/input/red-wine.csv'
red_wine = pd.read_csv(path_red_wine)
# print(red_wine.head())
# print(red_wine.shape) #(rows, columns)

##################################################################################################################

""" A SINGLE NEURON """

##################################################################################################################

""" STEP 1: INPUT SHAPE """

"""
How well can we predict a wine's perceived quality from the physiochemical measurements?

The target is 'quality', and the remaining columns are the features. How would you set the input_shape 
parameter for a Keras model on this task?
"""

# 11 feature columns, 1 target column --> 12 - 1 column
input_shape = [len(red_wine.columns) - 1]
# print(input_shape)

##################################################################################################################

""" STEP 2: DEFINE LINEAR MODEL """

"""
Now define a linear model appropriate for this task. Pay attention to how many inputs and outputs the model 
should have.
"""

# units --> no. of target column/s
# input_shape --> no. of feature columns
model = keras.Sequential([
    layers.Dense(units=1, input_shape=input_shape)
])


##################################################################################################################

""" STEP 3: LOOK AT THE WEIGHTS """

"""
Internally, Keras represents the weights of a neural network with tensors. Tensors are basically TensorFlow's 
version of a Numpy array with a few differences that make them better suited to deep learning. One of the most 
important is that tensors are compatible with GPU and TPU) accelerators. TPUs, in fact, are designed specifically 
for tensor computations.

A model's weights are kept in its weights attribute as a list of tensors. Get the weights of the model you defined 
above. 
"""

# check weight & bias
w, b = model.weights


##################################################################################################################

""" OPTIONAL: PLOT THE OUTPUTS OF AN UNTRAINED LINEAR MODEL """

"""
The kinds of problems we'll work on through Lesson 5 will be regression problems, where the goal is to predict 
some numeric target. Regression problems are like "curve-fitting" problems: we're trying to find a curve that 
best fits the data. Let's take a look at the "curve" produced by a linear model. (You've probably guessed that 
it's a line!)

We mentioned that before training a model's weights are set randomly. Run the cell below a few times to see the 
different lines produced with a random initialization. (There's no coding for this exercise -- it's just a 
demonstration.)
"""

model = keras.Sequential([
    layers.Dense(1, input_shape=[1]),
])

x = tf.linspace(-1.0, 1.0, 100)
y = model.predict(x)

plt.figure(dpi=100)
plt.plot(x, y, 'k')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("Input: x")
plt.ylabel("Target y")
w, b = model.weights # you could also use model.get_weights() here
plt.title("Weight: {:0.2f}\nBias: {:0.2f}".format(w[0][0], b[0]))
plt.show()

##################################################################################################################