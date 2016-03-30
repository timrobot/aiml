import numpy as np

def sigmoid(z):
  return 1. / (1. + np.exp(-z))

def grad_descent(X, y, theta, alpha, niter, lamb = 0):
  m = float(y.shape[0])
  costs = []
  for i in range(niter):

    theta = theta - alpha / m * (-y * )
    theta[1:] += alpha / m * lamb * theta[1:]
    costs.append(grad_cost(X, y, theta, lamb))

  return theta, costs

def grad_cost(X, y, theta, lamb = 0):
  m = float(y.shape[0])
  diff = X * theta - y
  variance = 0.5 / m * diff.T * diff
  variance[1:] += 0.5 / m * lamb * theta[1:].T * theta[1:]
  return variance

