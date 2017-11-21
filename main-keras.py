import numpy as np
import pandas as pd

# easy way to split data set into train/dev sets
from sklearn.model_selection import train_test_split

from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

def extract_data(data):
  # reshape the bands from flattened (5625,1) vectors to (75,75) vectors
  # note that all elements are floats
  band_1 = np.array([np.array(band).reshape(75, 75).astype("float32") for band in data["band_1"]])
  band_2 = np.array([np.array(band).reshape(75, 75).astype("float32") for band in data["band_2"]])
  print(np.shape(band_1))
  print(np.shape(band_2))
  is_iceberg = data["is_iceberg"]

  features = np.concatenate((
    band_1[:, :, :, np.newaxis],
    band_2[:, :, :, np.newaxis]
  ), axis=-1)
  labels = is_iceberg

  return features, labels

def build_model():
  model = Sequential()

  # input layer
  model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", input_shape=(75, 75, 2)))
  model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
  model.add(Dropout(0.2))

  model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Dropout(0.2))

  model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Dropout(0.2))

  model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Dropout(0.2))

  # flatten layers
  model.add(Flatten())

  # dense layer
  model.add(Dense(512))
  model.add(Activation("relu"))
  model.add(Dropout(0.2))

  # dense layer
  model.add(Dense(256))
  model.add(Activation("relu"))
  model.add(Dropout(0.2))

  # sigmoid layer
  model.add(Dense(1))
  model.add(Activation("sigmoid"))

  # compile model
  optimizer = Adam()
  model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

  # summary and return
  model.summary()
  return model

def main():
  raw_train = pd.read_json("data/train.json")
  raw_train_features, raw_train_labels = extract_data(raw_train)

  # split into train and dev sets
  train_features, dev_features, train_labels, dev_labels = train_test_split(
    raw_train_features, raw_train_labels, random_state=0, train_size=0.8)
  
  # construct model to use
  model = build_model()

  print(np.shape(train_features), np.shape(train_labels))

  # fit against the model
  model.fit(train_features, train_labels,
    batch_size=24, epochs=50, verbose=1,
    validation_data=(dev_features, dev_labels))

if __name__ == "__main__":
  main()
