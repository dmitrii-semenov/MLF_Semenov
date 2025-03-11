#%%

# Load all libraries used
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, InputLayer
import tensorflow as tf
import matplotlib.pyplot as plt

# Prepare data
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

# Creating a model
model = Sequential()
model.add(InputLayer(input_shape=(2)))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# Compile a model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train model
history = model.fit(X, y, epochs=2000, batch_size=1, verbose=0)
#%%