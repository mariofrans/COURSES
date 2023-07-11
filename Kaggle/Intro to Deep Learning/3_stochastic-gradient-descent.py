import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers
# from learntools.deep_learning_intro.dltools import animate_sgd

path_fuel = 'Jobs/Kaggle/Intro to Deep Learning/input/fuel.csv'
fuel = pd.read_csv(path_fuel)
# print(fuel.head())
# print(fuel.shape) #(rows, columns)

##################################################################################################################

""" STOCHASTIC GRADIENT DESCENT """

X = fuel.copy()
y = X.pop('FE') # remove target column

# preprocess all non-numerical data with encoding method
preprocessor = make_column_transformer(
    (
        StandardScaler(),
        make_column_selector(dtype_include=np.number)
    ),
    (
        OneHotEncoder(sparse=False),
        make_column_selector(dtype_include=object)
    ),
)

X = preprocessor.fit_transform(X)
y = np.log(y) # log transform target instead of standardizing

input_shape = [X.shape[1]]
# print("Input shape: {}".format(input_shape))

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),    
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])

##################################################################################################################

""" STEP 1: ADD LOSS & OPTIMIZER """

"""
Before training the network we need to define the loss and optimizer we'll use. Using the model's compile 
method, add the Adam optimizer and MAE loss.
"""

model.compile(
    optimizer='adam', 
    loss='mae', 
)

##################################################################################################################

""" STEP 2: TRAIN MODEL """

"""
Once you've defined the model and compiled it with a loss and optimizer you're ready for training. Train the 
network for 200 epochs with a batch size of 128. The input data is X with target y.
"""

X_train, X_valid, y_train, y_valid = train_test_split(X, y)

history = history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=128,
    epochs=200,
)

history_df = pd.DataFrame(history.history)
# start the plot at epoch 5 (changeable to get a different view)
history_df.loc[5:, ['loss']].plot()

##################################################################################################################

""" STEP 3: EVALUATE TRAINING """

"""
If you trained the model longer, would you expect the loss to decrease further?
"""

learning_rate = 0.05
batch_size = 32
num_examples = 256

# animate_sgd(
#     learning_rate=learning_rate,
#     batch_size=batch_size,
#     num_examples=num_examples,
#     # You can also change these, if you like
#     steps=50, # total training steps (batches seen)
#     true_w=3.0, # the slope of the data
#     true_b=2.0, # the bias of the data
# )

"""
With the learning rate and the batch size, you have some control over:
    1. How long it takes to train a model
    2. How noisy the learning curves are
    3. How small the loss becomes

To get a better understanding of these two parameters, we'll look at the linear model, our ppsimplest neural 
network. Having only a single weight and a bias, it's easier to see what effect a change of parameter has.

The next cell will generate an animation like the one in the tutorial. Change the values for learning_rate, 
batch_size, and num_examples (how many data points) and then run the cell. (It may take a moment or two.) Try 
the following combinations, or try some of your own:

learning_rate	batch_size	num_examples
0.05	        32	        256
0.05	        2	        256
0.05	        128	        256
0.02	        32	        256
0.2	            32	        256
1.0	            32	        256
0.9	            4096	    8192
0.99	        4096	    8192
"""

##################################################################################################################

""" STEP 4: LEARNING BATCH & SIZE """

"""
What effect did changing these parameters have? After you've thought about it, run the cell below for some 
discussion.

You probably saw that smaller batch sizes gave noisier weight updates and loss curves. This is because each 
batch is a small sample of data and smaller samples tend to give noisier estimates. Smaller batches can have 
an "averaging" effect though which can be beneficial.

Smaller learning rates make the updates smaller and the training takes longer to converge. Large learning 
rates can speed up training, but don't "settle in" to a minimum as well. When the learning rate is too large, 
the training can fail completely. (Try setting the learning rate to a large value like 0.99 to see this.)
"""

##################################################################################################################

