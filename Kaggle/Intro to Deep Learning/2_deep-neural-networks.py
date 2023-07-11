import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

path_concrete = 'Jobs/Kaggle/Intro to Deep Learning/input/concrete.csv'
concrete = pd.read_csv(path_concrete)
# print(concrete.head())
# print(concrete.shape) #(rows, columns)

##################################################################################################################

""" DEEP NEURAL NETWORKS """

##################################################################################################################

""" STEP 1: INPUT SHAPE """

# 8 feature columns, 1 target column --> 9 - 1 column
input_shape = [len(concrete.columns) - 1]
# print(input_shape)

##################################################################################################################

""" STEP 2: DEFINE A MODEL WITH HIDDEN LAYERS """

"""
Now create a model with three hidden layers, each having 512 units and the ReLU activation. Be sure to include 
an output layer of one unit and no activation, and also input_shape as an argument to the first layer.
"""

model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=512, activation='relu', input_shape=input_shape),
    layers.Dense(units=512, activation='relu', input_shape=input_shape),
    layers.Dense(units=512, activation='relu', input_shape=input_shape),
    # the linear output layer 
    layers.Dense(1),
])

##################################################################################################################

""" STEP 3: ACTIVATION LAYERS """

"""
Let's explore activations functions some.

The usual way of attaching an activation function to a Dense layer is to include it as part of the definition 
with the activation argument. Sometimes though you'll want to put some other layer between the Dense layer and 
its activation function. (We'll see an example of this in Lesson 5 with batch normalization.) In this case, we 
can define the activation in its own Activation layer, like so:

layers.Dense(units=8),
layers.Activation('relu')

This is completely equivalent to the ordinary way: layers.Dense(units=8, activation='relu').

Rewrite the following model so that each activation is in its own Activation layer.
"""

model = keras.Sequential([
    layers.Dense(32, input_shape=[8]),
    layers.Activation('relu'),
    layers.Dense(32),
    layers.Activation('relu'),
    layers.Dense(1)
])

##################################################################################################################

""" OPTIONAL: ALTERNATIVES TO RELU """

"""
There is a whole family of variants of the 'relu' activation -- 'elu', 'selu', and 'swish', among others -- all 
of which you can use in Keras. Sometimes one activation will perform better than another on a given task, so 
you could consider experimenting with activations as you develop a model. The ReLU activation tends to do well 
on most problems, so it's a good one to start with.

Let's look at the graphs of some of these. Change the activation from 'relu' to one of the others named above. 
Then run the cell to see the graph.
"""

# Change 'relu' to 'elu', 'selu', 'swish'... or something else
activation_layer = layers.Activation('relu')

x = tf.linspace(-3.0, 3.0, 100)
# once created, a layer is callable just like a function
y = activation_layer(x) 

plt.figure(dpi=100)
plt.plot(x, y)
plt.xlim(-3, 3)
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()

##################################################################################################################

