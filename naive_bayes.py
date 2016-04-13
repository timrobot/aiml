import numpy as np
import math
import sys
import dataset


def str2img(s):
    rows = len(filter(lambda s: s == "\n", s))
    cols = s.find("\n")
    img = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            img[i][j] = (s[i * (cols + 1) + j] == "#") \
                + 2 * (s[i * (cols + 1) + j] == "+")
    return img

def labels_estimate(labels):

#   Initialize a row of zeros. This will store the estimate of the prob.
#   of the different features occuring. This assumes that the prob. dist.
#   represented in integers >= 0

    labels_list = np.array(labels)
    prob_est = np.zeros(len(np.unique(labels_list)))
    for c in range(len(labels)):
        prob_est[labels[c]] += 1
    
#   Divides the prob. for each label by the total count of all labels 
    for i in range(len(prob_est)):
        prob_est[i] /= float(len(labels))

    return prob_est

def compute_jointprobtable(data, labels, prior_est):
    
#   Initializing the variables needed to make the joint prob. tables for
#   each feature and label

    entry_size = data[0].size
    num_labels = labels.size
    num_unique_marks = len(np.unique(np.array(data[0])))
    num_unique_labels = len(prior_est)
    num_unique_tables = num_unique_marks * num_unique_labels
    prob_table = [np.zeros(entry_size) for _ in xrange(num_unique_tables)]
    num_tables = len(prob_table)
    
#   Looping over the labels to compute joint prob. for each feature and
#   for each label
    for lab in range(num_labels):
        for entry in range(entry_size):
            #print "entry_size: ", entry_size
            #print int(num_unique_marks * labels[lab] + int(data[lab, entry]))
            prob_table[int(num_unique_marks * labels[lab] + int(data[lab, entry]))][entry] += 1

#    for j in range(30):
#        print prob_table[j]

    
    for tab in range(num_tables):
        for entry in range(entry_size):
            prob_table[tab][entry] /= prior_est[int(tab / num_unique_marks)]
            prob_table[tab][entry] /= num_labels
            if prob_table[tab][entry] > 0:
                prob_table[tab][entry] = math.log(prob_table[tab][entry])
            else:
                prob_table[tab][entry] = -10

#    print prob_table[2]
#    print prior_est

#    for i in range(30):
#        print prob_table[i]
    return prob_table

def prediction(data, data_labels, prior_est, prob_table):
    
    entry_size = data[0].size
    num_data_labels = data_labels.size
    num_unique_marks = len(np.unique(np.array(data[0])))
    num_unique_labels = len(prior_est)
    num_data = data.size / entry_size
    
    logprob_pred = [0.0 for i in range(num_data)]
    prediction = [0 for i in range(num_data)]
    tmp_prediction = [0.0, 0]

    for data_lab in range(num_data_labels):
        tmp_prediction = [-10000.0, 0]
        for label in range(num_unique_labels):
            tmp = prior_est[label]
            for entry in range(entry_size):
               #    Finding the log joint prob from the corresponding label and the feature type
               tmp += prob_table[int(num_unique_marks * label) + \
                    int(data[data_lab, entry])][entry]
#               print tmp
            if tmp_prediction[0] < tmp:
                tmp_prediction[0] = tmp
                tmp_prediction[1] = label
        logprob_pred[data_lab] = tmp_prediction[0]
        prediction[data_lab] = tmp_prediction[1]

#    print logprob_pred
#    print prediction
    return logprob_pred, prediction

def validation(prediction, actual):
    
    if len(prediction) != len(actual):
        print "The number of predictions do not match the actual number of entries"

    else:
        correct = 0
        for i in range(len(prediction)):
            if prediction[i] == actual[i]:
                correct += 1
    
        print float(correct) / float(len(prediction)) * 100


if __name__== "__main__":
    if len(sys.argv) < 5:
        print "Usage: python {0} datafile labelfile testfile testlabelfile".format(sys.argv[0])
        sys.exit(1)
    
    # Gets the data, labels and classes of the data and label files
    data, labels, classes = dataset.opendata(sys.argv[1], sys.argv[2])

    data = map(lambda s: str2img(s), data)
    data = map(lambda s: np.reshape(s, s.shape[0] * s.shape[1], 1), data)
    data = np.matrix(data)
    labels = np.matrix(labels).T

#   Prior Distribution
    param_est = labels_estimate(labels)

#   Log Joint Probability Tables
    joint_tables = compute_jointprobtable(data, labels, param_est)
    
#   Importing Test Data
    testdata, testlabels, testclasses = dataset.opendata(sys.argv[3], sys.argv[4])

    testdata = map(lambda s: str2img(s), testdata)
    testdata = map(lambda s: np.reshape(s, s.shape[0] * s.shape[1], 1), testdata)
    testdata = np.matrix(testdata)
    testlabels = np.matrix(testlabels).T

#   Prediction
    logprob, prediction = prediction(testdata, testlabels, param_est, joint_tables)

#   Validation
    validation(prediction, testlabels)
