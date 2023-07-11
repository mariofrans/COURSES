import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit

from tensorflow import keras
from tensorflow.keras import layers

path_spotify = 'Jobs/Kaggle/Intro to Deep Learning/input/spotify.csv'
spotify = pd.read_csv(path_spotify)
# print(spotify.head())
# print(spotify.shape) #(rows, columns)

path_concrete = 'Jobs/Kaggle/Intro to Deep Learning/input/concrete.csv'
concrete = pd.read_csv(path_concrete)
# print(concrete.head())
# print(concrete.shape) #(rows, columns)

##################################################################################################################

""" DROPOUT & BATCH NORMALIZATION """

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

def group_split(X, y, group, train_size=0.75):
    splitter = GroupShuffleSplit(train_size=train_size)
    train, test = next(splitter.split(X, y, groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])

X_train, X_valid, y_train, y_valid = group_split(X, y, artists)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
y_train = y_train / 100
y_valid = y_valid / 100

input_shape = [X_train.shape[1]]
# print("Input shape: {}".format(input_shape))

##################################################################################################################

""" STEP 1: ADD DROPOUT TO SPOTIFY MODEL """

"""
Here is the last model from Exercise 4. Add two dropout layers, one after the Dense layer with 128 units, and 
one after the Dense layer with 64 units. Set the dropout rate on both to 0.3.
"""

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
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
    verbose=0,
)

##################################################################################################################

""" STEP 2: EVALUATE DROPOUT """

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))

"""
Recall from Exercise 4 that this model tended to overfit the data around epoch 5. Did adding dropout seem 
to help prevent overfitting this time?
"""

"""
From the learning curves, you can see that the validation loss remains near a constant minimum even though 
the training loss continues to decrease. So we can see that adding dropout did prevent overfitting this time. 
Moreover, by making it harder for the network to fit spurious patterns, dropout may have encouraged the network 
to seek out more of the true patterns, possibly improving the validation loss some as well).
"""

##################################################################################################################

"""
Now, we'll switch topics to explore how batch normalization can fix problems in training.

Load the Concrete dataset. We won't do any standardization this time. This will make the effect of batch 
normalization much more apparent.
"""

df = concrete.copy()

df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)

X_train = df_train.drop('CompressiveStrength', axis=1)
X_valid = df_valid.drop('CompressiveStrength', axis=1)
y_train = df_train['CompressiveStrength']
y_valid = df_valid['CompressiveStrength']

input_shape = [X_train.shape[1]]

model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=input_shape),
    layers.Dense(512, activation='relu'),    
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])

model.compile(
    optimizer='sgd', # SGD is more sensitive to differences of scale
    loss='mae',
    metrics=['mae'],
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=64,
    epochs=100,
    verbose=0,
)

history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
print(("Minimum Validation Loss: {:0.4f}").format(history_df['val_loss'].min()))

"""
Did you end up with a blank graph? Trying to train this network on this dataset will usually fail. Even 
when it does converge (due to a lucky weight initialization), it tends to converge to a very large number.
"""

##################################################################################################################

""" STEP 3: ADD BATCH NORMALIZATION LAYERS """

"""
Batch normalization can help correct problems like this.

Add four BatchNormalization layers, one before each of the dense layers. (Remember to move the input_shape 
argument to the new first layer.)
"""

model = keras.Sequential([
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu', input_shape=input_shape),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1),
])

model.compile(
    optimizer='sgd',
    loss='mae',
    metrics=['mae'],
)

EPOCHS = 100
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=64,
    epochs=EPOCHS,
    verbose=0,
)

##################################################################################################################

""" STEP 4: EVALUATE BATCH NORMALIZATION """

history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
print(("Minimum Validation Loss: {:0.4f}").format(history_df['val_loss'].min()))

"""
Did adding batch normalization help?
"""

"""
You can see that adding batch normalization was a big improvement on the first attempt! By adaptively scaling 
the data as it passes through the network, batch normalization can let you train models on difficult datasets.
"""

##################################################################################################################

