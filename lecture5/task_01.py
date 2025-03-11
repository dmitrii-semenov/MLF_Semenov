#%%

# Load all libraries used
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Prepare data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Creating a model
model = Sequential()
model.add(InputLayer(input_shape=(2,)))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# Compile a model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train model
history = model.fit(X, y, epochs=2000, batch_size=1, verbose=0)

# Model evaluation
loss, accuracy = model.evaluate(X, y, verbose=0)
print('Accuracy: {:.2f}'.format(accuracy*100))

# Model prediction
for id_x, data_sample in enumerate(X):
  prediction = model.predict(np.array(data_sample).reshape(1, -1))
  print(f"Data sample is {data_sample}, prediction from model {prediction}, ground_truth {y[id_x]}")

# Display loss function during the training process and acuracy
plt.figure()
plt.plot(history.history['loss'])
plt.xlabel('n epochs')
plt.ylabel('loss')
plt.show()
#%%