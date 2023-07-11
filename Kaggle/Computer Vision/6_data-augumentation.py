import os, warnings
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

path_train = '../input/car-or-truck/train'
path_valid = '../input/car-or-truck/valid'

##################################################################################################################

""" DATA AUGUMENTATION """

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

""" EXPLORE AUGUMENTATION """

# all of the "factor" parameters indicate a percent-change
augment = keras.Sequential([
#     preprocessing.RandomContrast(factor=0.5),
#     preprocessing.RandomFlip(mode='horizontal'), # meaning, left-to-right
#     preprocessing.RandomFlip(mode='vertical'), # meaning, top-to-bottom
#     preprocessing.RandomWidth(factor=0.15), # horizontal stretch
#     preprocessing.RandomRotation(factor=0.20),
    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
])


ex = next(iter(ds_train.unbatch().map(lambda x, y: x).batch(1)))
plt.figure(figsize=(10,10))

for i in range(16):
    image = augment(ex, training=True)
    plt.subplot(4, 4, i+1)
    plt.imshow(tf.squeeze(image))
    plt.axis('off')

plt.show()

"""
Do the transformations you chose seem reasonable for the Car or Truck dataset?
"""

##################################################################################################################

""" STEP 1: EURO SAT """

"""
In this exercise, we'll look at a few datasets and think about what kind of augmentation might be appropriate. 
Your reasoning might be different that what we discuss in the solution. That's okay. The point of these problems 
is just to think about how a transformation might interact with a classification problem -- for better or worse.
"""

"""
The EuroSAT dataset consists of satellite images of the Earth classified by geographic feature. 
Below are a number of images from this dataset.

What kinds of transformations might be appropriate for this dataset?
"""

"""
It seems to this author that flips and rotations would be worth trying first since there's no concept of 
orientation for pictures taken straight overhead. None of the transformations seem likely to confuse classes, 
however.
"""

##################################################################################################################

""" STEP 2: TENSORFLOW FLOWERS """

"""
The TensorFlow Flowers dataset consists of photographs of flowers of several species. Below is a sample.

What kinds of transformations might be appropriate for the TensorFlow Flowers dataset?
"""

"""
It seems to this author that horizontal flips and moderate rotations would be worth trying first. Some 
augmentation libraries include transformations of hue (like red to blue). Since the color of a flower seems 
distinctive of its class, a change of hue might be less successful. On the other hand, there is suprising 
variety in cultivated flowers like roses, so, depending on the dataset, this might be an improvement after all!
"""

##################################################################################################################

""" STEP 3: ADD PREPROCESSING LAYERS """

"""
Now you'll use data augmentation with a custom convnet similar to the one you built in Exercise 5. Since data 
augmentation effectively increases the size of the dataset, we can increase the capacity of the model in turn 
without as much risk of overfitting.
"""

model = keras.Sequential([
    layers.InputLayer(input_shape=[128, 128, 3]),
    
    # Data Augmentation
    preprocessing.RandomContrast(factor=0.10),
    preprocessing.RandomFlip(mode='horizontal'),
    preprocessing.RandomRotation(factor=0.10),

    # Block One
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Block Two
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Block Three
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Head
    layers.BatchNormalization(renorm=True),
    layers.Flatten(),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

# train the model
optimizer = tf.keras.optimizers.Adam(epsilon=0.01)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=50,
)

##################################################################################################################

""" STEP 4: EXAMINE TRAINED MODEL """

# Plot learning curves
import pandas as pd
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()

"""
Examine the training curves. What there any sign of overfitting? How does the performance of this model compare 
to other models you've trained in this course?
"""

"""
The learning curves in this model stayed close together for much longer than in previous models. This suggests 
that the augmentation helped prevent overfitting, allowing the model to continue improving.

And notice that this model achieved the highest accuracy of all the models in the course! This won't always be 
the case, but it shows that a well-designed custom convnet can sometimes perform as well or better than a much 
larger pretrained model. Depending on your application, having a smaller model (which requires fewer resources) 
could be a big advantage.
"""

##################################################################################################################