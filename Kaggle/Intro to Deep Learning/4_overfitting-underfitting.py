import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

path_spotify = 'Jobs/Kaggle/Intro to Deep Learning/input/spotify.csv'
spotify = pd.read_csv(path_spotify)
# print(spotify.head())
# print(spotify.shape) #(rows, columns)

##################################################################################################################

""" OVERFITTING & UNDERFITTING """

X = spotify.copy().dropna()
y = X.pop('track_popularity')
artists = X['track_artist']

features_num = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms']
features_cat = ['playlist_genre']

preprocessor = make_column_transformer(
    (StandardScaler(), features_num),
    (OneHotEncoder(), features_cat),
)

# We'll do a "grouped" split to keep all of an artist's songs in one
# split or the other. This is to help prevent signal leakage.
def group_split(X, y, group, train_size=0.75):
    splitter = GroupShuffleSplit(train_size=train_size)
    train, test = next(splitter.split(X, y, groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])

X_train, X_valid, y_train, y_valid = group_split(X, y, artists)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
y_train = y_train / 100 # popularity is on a scale 0-100, so this rescales to 0-1.
y_valid = y_valid / 100

input_shape = [X_train.shape[1]]
# print("Input shape: {}".format(input_shape))

##################################################################################################################

""" LINEAR MODEL """

model = keras.Sequential([
    layers.Dense(1, input_shape=input_shape),
])

model.compile(
    optimizer='adam',
    loss='mae',
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
    verbose=0, # suppress output since we'll plot the curves
)

##################################################################################################################

""" STEP 1: EVALUATE BASELINE """

# Start the plot at epoch 0
history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))

# Start the plot at epoch 10
history_df.loc[10:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))

"""
What do you think? Would you say this model is underfitting, overfitting, just right?

The gap between these curves is quite small and the validation loss never increases, so it's more likely that 
the network is underfitting than overfitting. It would be worth experimenting with more capacity to see if 
that's the case.
"""

##################################################################################################################

""" STEP 2: ADD CAPACITY """

"""
Now let's add some capacity to our network. We'll add three hidden layers with 128 units each. Run the next 
cell to train the network and see the learning curves.
"""

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mae',
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))

"""
What is your evaluation of these curves? Underfitting, overfitting, just right?
"""

"""
Now the validation loss begins to rise very early, while the training loss continues to decrease. This 
indicates that the network has begun to overfit. At this point, we would need to try something to prevent 
it, either by reducing the number of units or through a method like early stopping.
"""

##################################################################################################################

""" STEP 3: DEFINE EARLY STOPPING CALLBACK """

"""
Now define an early stopping callback that waits 5 epochs (patience') for a change in validation loss of at 
least 0.001 (min_delta) and keeps the weights with the best loss (restore_best_weights).
"""

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,    # minimium amount of change to count as an improvement
    patience=5,         # how many epochs to wait before stopping
    restore_best_weights=True,
)

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),    
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mae',
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
    callbacks=[early_stopping]
)

##################################################################################################################

""" STEP 4: TRAIN & INTERPRET """

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))

"""
Was this an improvement compared to training without early stopping?
"""

"""
The early stopping callback did stop the training once the network began overfitting. Moreover, by including 
restore_best_weights we still get to keep the model where validation loss was lowest.

If you like, try experimenting with patience and min_delta to see what difference it might make.
"""

##################################################################################################################

