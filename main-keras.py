import json
import numpy as np

from sklearn.model_selection import train_test_split

def read_data(filename):
  data = json.load(open(filename))
  
  # TODO: use a single pass so this isn't doing multiple n loops
  # extract raw features from the data
  band_1 = np.array([sample["band_1"] for sample in data])
  band_2 = np.array([sample["band_2"] for sample in data])
  labels = np.array([sample["is_iceberg"] for sample in data])
  
  # Combine features lengthwise
  features = np.hstack((band_1, band_2))

  return train_test_split(features, labels, test_size=0.1, random_state=RANDOM_STATE)

def main():
  x_train, x_test, y_train, y_test = read_data("data/train.json")
  ### TODO: everything here

if __name__ == "__main__":
  main()
