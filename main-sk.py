import json
import numpy as np

from sklearn import metrics, svm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

RANDOM_STATE = 0

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

def train_svm(features, labels):
  classifier = svm.SVC(gamma = 0.001)
  classifier.fit(features, labels)
  return classifier

def train_kmeans(features, labels):
  classifier = KMeans(n_clusters=2, random_state=RANDOM_STATE)
  classifier.fit(features)
  return classifier

def main():
  x_train, x_test, y_train, y_test = read_data("data/train.json")
  
  # Uncomment out only one of these classifiers
  # classifier = train_svm(x_train, y_train) # SVM
  classifier = train_kmeans(x_train, y_train) # KMeans

  # run classifier on held out set
  y_pred = classifier.predict(x_test)
  
  # some debugging information
  print("Classification report:\n", metrics.classification_report(y_test, y_pred))
  print("Confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
  main()
