import dataset, sys
import numpy as np

data, labels, n = dataset.opendata(sys.argv[1], sys.argv[2])

print len(data), n

print data[0], labels[0]
