# Common
!pip install opencv-python
import cv2
import os 
import keras
import numpy as np
import tensorflow as tf

# Data 
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator 
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# Data Visualization 
import plotly.express as px
import matplotlib.pyplot as plt


# Model 
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Dense, GlobalAvgPool2D

# Callbacks 
from keras.callbacks import EarlyStopping, ModelCheckpoint

%%time

# Specify Data Path
file_path = 'A_Z Handwritten Data.csv'

# Column Names
names = ['class']
for id in range(1,785):
    names.append(id)

# Load Data
df = pd.read_csv(file_path,header=None, names=names)
df.head()

class_mapping = {}
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
for i in range(len(alphabets)):
    class_mapping[i] = alphabets[i]
class_mapping

df['class'].map(class_mapping).unique()
df['class']

names = df['class'].value_counts().keys().map(class_mapping)
values = df['class'].value_counts()

# Plot Class Distribution
fig = px.pie(
    names=names,
    values=values,
    height=800,
    title='Class Distribution'
)
fig.update_layout({'title':{'x':0.5}})
fig.show()

# Plot Class Distribution
fig = px.bar(
    x=names,
    y=values,
    height=800,
    title='Class Distribution'
)
fig.update_layout({'title':{'x':0.5}})
fig.show()

y_full = df.pop('class')
x_full = df.to_numpy().reshape(-1,28,28, 1)

splitter = StratifiedShuffleSplit(n_splits=3,test_size=0.2)
for train_ids, test_ids in splitter.split(x_full, y_full):
    X_train_full, y_train_full = x_full[train_ids], y_full[train_ids].to_numpy()
    X_test, y_test = x_full[test_ids], y_full[test_ids].to_numpy()
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.1)
    
    plt.figure(figsize=(15,8))
for i in range(1, 11):
    
    id = np.random.randint(len(X_train))
    image, label = tf.squeeze(X_train[id]), class_mapping[int(y_train[id])]
    
    plt.subplot(2,5,i)
    plt.imshow(image, cmap='binary')
    plt.title(label)
    plt.axis('off')
    
plt.tight_layout()
plt.show()

model = Sequential([
    Conv2D(32, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal', input_shape=(28, 28, 1)),
    MaxPool2D(),

    BatchNormalization(),
    Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal'),
    BatchNormalization(),
    Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal'),
    MaxPool2D(),

    BatchNormalization(),
    Conv2D(128, kernel_size=3, padding='same', kernel_initializer='he_normal'),
    BatchNormalization(),
    Conv2D(128, kernel_size=3, padding='same', kernel_initializer='he_normal'),
    MaxPool2D(),

    BatchNormalization(),
    Conv2D(256, kernel_size=3, padding='same', kernel_initializer='he_normal'),
    BatchNormalization(),
    Conv2D(256, kernel_size=3, padding='same', kernel_initializer='he_normal'),
  
    GlobalAvgPool2D(),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(26, activation='sigmoid')
])

# Compile
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Callbacks
cbs = [EarlyStopping(patience=3, restore_best_weights=True), ModelCheckpoint("Model-v1.h5", save_best_only=True)]

# Training
model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=15,
    callbacks=cbs
)

model.evaluate(X_test,y_test)

plt.figure(figsize=(20,20))
for i in range(1, 101):
    
    id = np.random.randint(len(X_test))
    image, label = X_test[id].reshape(28,28), class_mapping[int(y_test[id])]
    pred = class_mapping[int(np.argmax(model.predict(image.reshape(-1,28,28,1))))]
    
    plt.subplot(10,10,i)
    plt.imshow(image, cmap='binary')
    plt.title(f"Org: {label}, Pred: {pred}")
    plt.axis('off')
    
plt.tight_layout()
plt.show()

image = cv2.imread("o.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image = cv2.resize(image, (28,28))
print("Before Line 93, image size is: " , image.shape)
image = image.reshape(-1,28,28,1)
image = image / 255.

# make a prediction using the trained model
predictions = model.predict(image)

# select the class with the highest probability
predicted_class = np.argmax(predictions)

print(predictions)
print(predicted_class+1)
print("This is equivalent to:", class_mapping[predicted_class + 1])
