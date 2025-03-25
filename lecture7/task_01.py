#%%

# Import libraries
from tensorflow import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)

# Display random images function
def display_random_images(x_data: np.array, y_data: np.array, count: int = 10) -> None:
  index = np.array(len(x_data))
  selected_ind = np.random.choice(index, count)

  selected_img = x_data[selected_ind]
  selected_labels = y_data[selected_ind]
  concat_img = np.concatenate(selected_img, axis=1)

  plt.figure(figsize=(20,10))
  plt.imshow(concat_img, cmap="gray")

  for id_label, label in enumerate(selected_labels):
    plt.text(14 + 28*id_label, 28*(5/4), label)
  plt.axis('off')
  plt.show()

# Prepare data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape,X_test.shape)

display_random_images(X_train, y_train)

# Dataset preprocessing
X_train = X_train.astype('float32') / 255.0
X_train = X_train.reshape(-1,28,28,1)
y_train = to_categorical(y_train, num_classes=10)

X_test = X_test.astype('float32') / 255.0
X_test = X_test.reshape(-1,28,28,1)
y_test = to_categorical(y_test, num_classes=10)

# Define the model structure
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
#model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.005)))
#model.add(Dense(10, activation='softmax', kernel_regularizer=l2(0.005)))
#model.add(Dropout(0.1))

# Check model description 
print(model.summary())

# Compile model
optimizer = Adam(learning_rate = 0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split = 0.2)

#%%
# Model evaluation on validation data
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Epoch number, [-]")
plt.ylabel("Loss, [-]")
plt.legend(["loss", "val_loss"])
plt.show()

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel("Epoch number, [-]")
plt.ylabel("Accuracy, [-]")
plt.legend(["accuracy", "val_accuracy"])
plt.show()

# Model evaluation on test data
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print(f'Test accuracy: {score[1]*100} %')

# Generate predictions on test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - MNIST Test Data')
plt.show()

# Tested with dropout 0.1 and 0.2 => accuracy is around 80-90%, BUT validation accuracy is still around 98.5%, better without it
# Added EarlyStopping with patience 3 => accuracy is nearly the same, prevent overfitting
# Tried to add 0.01 and 0.005 L2 regularization technics to the Dense layers => validation accuracy decreased to 94.25% and 95.92% respectively
#%%