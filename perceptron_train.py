import numpy as np # used for the matrix calc's
import dataset
import sys

def normal_eqn(X, y):
  theta = np.linalg.pinv(X.T * X) * X.T * y
  return theta

def cost(X, y, theta):
  m = float(y.shape[0])
  diff = X * theta - y
  variance = 0.5 / m * diff.T * diff
  return variance

def grad_descent(X, y, theta, alpha, niter, lamb = 0):
  m = float(y.shape[0])
  costs = []
  for i in range(niter):
    theta = theta - alpha / m * X.T * (X * theta - y)
    theta[1:] += alpha / m * lamb * theta[1:]
    costs.append(grad_cost(X, y, theta, lamb))
  return theta, costs

def grad_cost(X, y, theta, lamb = 0):
  m = float(y.shape[0])
  diff = X * theta - y
  variance = 0.5 / m * diff.T * diff
  variance[1:] += 0.5 / m * lamb * theta[1:].T * theta[1:]
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
  m = float(y.shape[0])
  conf = ((X * thetas).argmax(axis=1) == y).sum() / m
  return conf, 1-conf

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print "Usage: python {0} datafile labelfile".format(sys.argv[0])
    sys.exit(1)
  data, labels, classes = dataset.opendata(sys.argv[1], sys.argv[2])
  data = map(lambda s: str2img(s), data)
  data = map(lambda s: np.reshape(s, s.shape[0] * s.shape[1], 1), data)
  data = np.matrix(data)
  labels = np.matrix(labels).T

  # do one vs all classification for all the classes
  thetas = np.matrix([[]])
  for i in range(classes):
    theta, _ = linreg_train(data, labels == i)
    if not thetas.any():
      thetas = theta
    else:
      thetas = np.concatenate((thetas, theta), axis=1)

  # check the result
  conf, err = class_err(data, labels, thetas)
  print "Accuracy: {0}%".format(conf * 100)
