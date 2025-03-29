#%%

# Import all libraries used
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# define a function for train and test split
def train_test_split_custom(pd_data: pd.DataFrame, test_ratio: float = 0.2) -> tuple:
    pd_dataset = pd_data.copy()
    pd_dataset = pd_dataset[pd_dataset.columns[1:]]
    index = np.arange(len(pd_dataset))
    index = np.random.permutation(index)
    train_ammount = int(len(index)*test_ratio)
    train_ids = index[train_ammount:]
    test_ids = index[:train_ammount]
    
    train_dataset = pd_dataset[pd_dataset.index.isin(train_ids)].reset_index()
    test_dataset = pd_dataset[pd_dataset.index.isin(test_ids)].reset_index()
    
    train_dataset = train_dataset[train_dataset.columns[1:]]
    test_dataset = test_dataset[test_dataset.columns[1:]]

    return train_dataset[train_dataset.columns[1:]], train_dataset[train_dataset.columns[0]], test_dataset[test_dataset.columns[1:]], test_dataset[test_dataset.columns[0]]

# Loading dataset
path_to_dataset = 'data/voting_complete.csv'
pd_dataset = pd.read_csv(path_to_dataset)

# Train/Test Split
x_train, y_train, x_test, y_test = train_test_split_custom(pd_dataset)

# Replace missing values with NaN
x_train.replace('?', np.nan, inplace=True)
x_train.fillna(x_train.mode().iloc[0],inplace=True)
x = pd.get_dummies(x_train)

# Replace categories with '0' or '1'
y = y_train.replace({'republican': 1,'democrat': 0})

# Create the model
model = Sequential()
model.add(Dense(2, input_shape = (x.shape[1],), activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))

# Model summary
print(model.summary())

# Compile a model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train model
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
history = model.fit(X_train, y_train, 
                    epochs=40, batch_size=4, 
                    verbose=1, validation_data = (X_val,y_val))

# Preprocessing for validation
x_test.replace('?', np.nan, inplace=True)
x_test.fillna(x_test.mode().iloc[0],inplace=True)
x_te = pd.get_dummies(x_test)
y_te = y_test.replace({'republican': 1,'democrat': 0})

# Model evaluation
loss, accuracy = model.evaluate(x_te, y_te, verbose=0)
print('Tested accuracy: {:.2f}'.format(accuracy*100))
print('Tested loss: {:.2f}'.format(loss*100))

# Display loss function during the training process and acuracy
plt.figure()
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'],c = 'red',label = 'validation loss')
plt.xlabel('n epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Display accuracy function during the training process and acuracy
plt.figure()
plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'],c = 'red', label = 'validation accuracy')
plt.xlabel('n epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
#%%