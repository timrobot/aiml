import sys
from dataset import *

def str2img(s):
  rows = len(filter(lambda s: s == "\n", s))
  cols = s.find("\n")
  img = np.zeros((rows, cols))
  for i in range(rows):
    for j in range(cols):
      img[i][j] = (s[i*(cols+1)+j] == "#") + 0.5 * (s[i*(cols+1)+j] == "+")
  return img

if __name__ == "__main__":
  if len(sys.argv) != 5:
    print "Usage: {0} datain labelin dataout labelout".format(sys.argv[0])
    sys.exit(1)

  X, y, n = opendata(sys.argv[1], sys.argv[2])
  X = map(lambda s: str2img(s), X)
  X = map(lambda s: np.reshape(s, s.shape[0] * s.shape[1], 1), X)
  X = np.matrix(X)
  y = np.matrix(y).T

  datafile = open(sys.argv[3], "w")
  datafile.write("[{0},{1}]\n".format(X.shape[0], X.shape[1]))
  for i in range(X.shape[0]):
    for j in range(X.shape[1]):
      datafile.write("{0}\n".format(X[i][j]))
  datafile.close()
  labelfile = open(sys.argv[4], "w")
  labelfile.write("[{0},1]\n".format(X.shape[0]))
  for i in range(X.shape[0]):
    labelfile.write("{0}\n".format(y[i][0]))
  labelfile.close()
