import sys
import os
import numpy as np

def numlines(fname):
  """ Get the number of lines in a file
      @param fname the name of the rfile
      @return the number of lines in the file
  """
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i + 1

def opendata(datafile, labelfile):
  """ Get the dataset given two file names
      @param datafile the name of the file containing the data
      @param labelfile the name of the file containing the labels
      @return a list of [data, label, number of classes]
  """
  datafp = open(datafile, "r")
  labelsfp = open(labelfile, "r")

  datalines = numlines(datafile)
  labellines = numlines(labelfile)
  imglines = datalines / labellines

  data = []
  labels = []
  for i in range(labellines):
    image = ""
    for j in range(imglines):
      image += datafp.readline()
    data.append(image)
    label = int(labelsfp.readline().replace("\n", ""))
    labels.append(label)

  return data, labels, len(set(labels))
