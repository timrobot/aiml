import numpy as np # used for the matrix calc's
import dataset
import sys

def normal_eqn(X, y):
  theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)
  return theta

def cost(X, y, theta):
  m = float(X.shape[0])
  diff = np.dot(X, theta) - y
  variance = 0.5 / m * np.dot(diff.T, diff)
  return variance

def linreg_train(X, y):
  # first try out the normal equation, see how it works out
  theta = normal_eqn(X, y)
  err = cost(X, y, theta)
  return theta, err

def str2img(s):
  rows = len(filter(lambda s: s == "\n", s))
  cols = s.find("\n")
  img = np.zeros((rows, cols))
  for i in range(rows):
    for j in range(cols):
      img[i][j] = (s[i*(cols+1)+j] == "#") + 0.5 * (s[i*(cols+1)+j] == "+")
  return img

def class_err(X, y, thetas):
  m = float(X.shape[0])
  k = np.dot(X, thetas).argmax(axis=1).astype(int)
  Y = y.astype(int)
  conf = 0
  for i in range(int(m)):
    if k[i] == Y[i][0]:
      conf += 1
  conf /= m
  return conf, 1 - conf

import scipy.io as sio

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print "Usage: python {0} datafile labelfile".format(sys.argv[0])
    sys.exit(1)
  X, y, classes = dataset.opendata(sys.argv[1], sys.argv[2])
  X = map(lambda s: str2img(s), X)
  X = map(lambda s: np.reshape(s, s.shape[0] * s.shape[1], 1), X)
  X = np.matrix(X)
  y = np.matrix(y).T

  # do one vs all classification for all the classes
  X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
  thetas = np.matrix([[]])
  for i in range(classes):
    theta, _ = linreg_train(X, y == i)
    if not thetas.any():
      thetas = theta
    else:
      thetas = np.concatenate((thetas, theta), axis=1)

  # check the result
  conf, err = class_err(X, y, thetas)
  print "Accuracy: {0}%".format(conf * 100)
