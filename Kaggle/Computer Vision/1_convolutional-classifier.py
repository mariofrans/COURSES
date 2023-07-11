import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers

path_train = '../input/car-or-truck/train'
path_valid = '../input/car-or-truck/valid'
path_models = '../input/cv-course-models/cv-course-models/inceptionv1'

##################################################################################################################

""" CONVOLUTIONAL CLASSIFIER """

# Reproducability
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seed()

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore") # to clean up output cells

# Load training and validation sets
ds_train_ = image_dataset_from_directory(
    path_train,
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)

ds_valid_ = image_dataset_from_directory(
    path_valid,
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
)

# Data Pipeline
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = (
    ds_train_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

ds_valid = (
    ds_valid_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

pretrained_base = tf.keras.models.load_model(path_models)

##################################################################################################################

""" STEP 1: DEFINE PRETRAINED BASE """

"""
Now that you have a pretrained base to do our feature extraction, decide whether this base should be trainable 
or not.
"""

pretrained_base.trainable = False

"""
When doing transfer learning, it's generally not a good idea to retrain the entire base -- at least not without 
some care. The reason is that the random weights in the head will initially create large gradient updates, which 
propogate back into the base layers and destroy much of the pretraining. Using techniques known as fine tuning 
it's possible to further train the base on new data, but this requires some care to do well.
"""

##################################################################################################################

""" STEP 2: ATTACH HEAD """

model = keras.Sequential([
    pretrained_base,
    layers.Flatten(),
    layers.Dense(units=6, activation='relu'),
    layers.Dense(units=1, activation='sigmoid')
])

##################################################################################################################

""" STEP 3: TRAIN """

"""
Before training a model in Keras, you need to specify an optimizer to perform the gradient descent, a loss 
function to be minimized, and (optionally) any performance metrics. The optimization algorithm we'll use for 
this course is called "Adam", which generally performs well regardless of what kind of problem you're trying 
to solve.

The loss and the metrics, however, need to match the kind of problem you're trying to solve. Our problem is 
a binary classification problem: Car coded as 0, and Truck coded as 1. Choose an appropriate loss and an 
appropriate accuracy metric for binary classification
"""

optimizer = tf.keras.optimizers.Adam(epsilon=0.01)

model.compile(
    optimizer=optimizer,
    loss = 'binary_crossentropy',
    metrics=['binary_accuracy'],
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=30,
)

##################################################################################################################

""" STEP 4: EXAMINE LOSS ACCURACY """

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()

"""
Do you notice a difference between these learning curves and the curves for VGG16 from the tutorial? What does 
this difference tell you about what this model (InceptionV2) learned compared to VGG16? Are there ways in which 
one is better than the other? Worse?
"""

"""
That the training loss and validation loss stay fairly close is evidence that the model isn't just memorizing 
the training data, but rather learning general properties of the two classes. But, because this model converges 
at a loss greater than the VGG16 model, it's likely that it is underfitting some, and could benefit from some 
extra capacity.
"""

##################################################################################################################

