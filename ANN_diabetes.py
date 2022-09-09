#%%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf

#%%
# Load data
data = pd.read_csv('diabetes.csv')
data.head()

#%%
# Data summary
data.describe()


# Data scaling
X_scaled = MinMaxScaler().fit_transform(data.iloc[:,0:8])
y = data['Outcome'].to_numpy()

# Data spliting
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=500, random_state=123)

## Implement ANN architecture

model_1 = tf.keras.models.Sequential()
model_1.add(tf.keras.layers.Dense(units=20, activation='relu',input_shape = [8]))
model_1.add(tf.keras.layers.Dense(units=10, activation='relu'))
model_1.add(tf.keras.layers.Dense(units=1, activation='relu'))

model_1.summary()
model_1.compile(
    loss='binary_crossentropy',
    optimizer= "adam",
    metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
)

history = model_1.fit(X_train, y_train, epochs=50,batch_size = 5,validation_split = 0.2)

plt.figure(figsize=(6, 7))

# Model accuracy
plt.subplot(2, 1, 1)
plt.plot(history.history['val_accuracy'],'-b')
plt.plot(history.history['accuracy'],'-g')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])

# Model loss
plt.subplot(2, 1, 2)
plt.plot(history.history['val_loss'],'-b')
plt.plot(history.history['loss'],'-g')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

# Prediction
y_pred = model_1.predict(X_test)
test_loss, test_acc = model_1.evaluate(X_test, y_test)

print('Accuracy = ', test_acc)

# %%
