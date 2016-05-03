#!/usr/bin/env python2
import scipy.io as sio
import sys

if len(sys.argv) != 4:
  print "usage: {0} input_mat output_data output_label".format(sys.argv[0])
  sys.exit(1)

testdata = sio.loadmat(sys.argv[1])
data = testdata["X"]
labels = testdata["y"]
labels %= 10
classes = 10

fp = open(sys.argv[2], "w")
fp.write("[{0},{1}]\n".format(data.shape[0], data.shape[1]))
for i in range(data.shape[0]):
  for j in range(data.shape[1]):
    fp.write("{0}\n".format(data[i][j]))
fp.close()
fp = open(sys.argv[3], "w")
fp.write("[{0},1]\n".format(labels.shape[0]))
for i in range(labels.shape[0]):
  fp.write("{0}\n".format(labels[i][0]))
fp.close()
