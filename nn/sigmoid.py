import numpy as np

def sigmoid(z):
  return 1. / (1. + np.exp(-z))

def one_step_learn(x, y):
  w = (x.t() * x).i() * x.t() * y.t() # this is the one step learning function
