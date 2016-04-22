import numpy as np
import math
import sys
import random
import dataset
import matplotlib.pyplot as plt

lap_smooth = 0


def str2img(s):
    rows = len(filter(lambda s: s == "\n", s))
    cols = s.find("\n")
    img = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            img[i][j] = (s[i * (cols + 1) + j] == "#") \
                + (s[i * (cols + 1) + j] == "+")
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


def convolve(X, A):
    return X[0,0]*A[0,0] + X[0,1]*A[0,1] + X[0,2]*A[0,2] + X[1,0]*A[1,0] + X[1,1]*A[1,1] + X[1,2]*A[1,2] + X[2,0]*A[2,0] + X[2,1]*A[2,1] + X[2,2]*A[2,2]

def sobel_filter(data, img_dim):

    entry_size = data[0].size
    num_labels = data.size / entry_size
    (img_row, img_col) = img_dim
    entry_shape = (1, (img_dim[0] - 2) * (img_dim[1] - 2))

    X = np.matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Y = np.matrix([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    new_data = []

    for lab in range(num_labels):
        #print "\tWorking on {0} out of {1}", lab, num_label
        new_entry = np.zeros(entry_shape)
        count = 0
        for i in range(img_row - 2):
            for j in range(img_col - 2):
                
                A = np.matrix([[data[lab, (img_col * i) + j], data[lab, (img_col * i) + (j + 1)], data[lab, (img_col * i) + (j + 2)]], \
                               [data[lab, (img_col * (i + 1)) + j], data[lab, (img_col * (i + 1)) + (j + 1)], data[lab, (img_col * (i + 1)) + (j + 2)]], \
                               [data[lab, (img_col * (i + 2)) + j], data[lab, (img_col * (i + 2)) + (j + 1)], data[lab, (img_col * (i + 2)) + (j + 2)]]])

                Gx = convolve(X, A) 
                Gy = convolve(Y, A)
                D = np.arctan2(Gy, Gx)
                new_entry[0, count] = D
                count += 1
        new_data.append(new_entry)

    new_data = map(lambda s: np.reshape(s, s.shape[0] * s.shape[1], 1), new_data)
    return np.matrix(new_data)
                


def compute_jointprobtable(data, labels, prior_est):
    
#   Initializing the variables needed to make the joint prob. tables for
#   each feature and label

    entry_size = data[0].size                                               # Getting the size of the image
    num_labels = labels.size                                                # Getting the number of images
#    num_unique_marks = len(np.unique(np.array(data[0])))                    # Getting the number of unique markers (" ", "#", "+") <=> (0, 1, 2)
    num_unique_marks = 20
    num_unique_labels = len(prior_est)                                      # Getting the number of unique labels
    num_unique_tables = num_unique_marks * num_unique_labels                # Calculating the number of unique tables
    prob_table = [np.zeros(entry_size) for _ in xrange(num_unique_tables)]  # Initializing the probability tables, for each unique label and marker
    num_tables = len(prob_table)                                            # Getting the number of tabels

#    print data[0]
#    print data[1]
#    print data[2]

#   Looping over the labels to compute joint prob. for each feature and
#   for each label.

    for lab in range(num_labels):
        for entry in range(entry_size):
            
            # For each element in the current image, increment the table associated with the 
            # corresponding label and feature by one to count the occurrence

#            print lab, entry, len(data), entry_size, data[0].size, labels.size
#            print data[lab,entry], entry
            prob_table[int(num_unique_marks * labels[lab] + int(data[lab, entry]))][entry] += 1

#    for j in range(30):
#        print prob_table[j]



#   Looping over the computed prob. tables and each entry to compute conditional probabilities

    for tab in range(num_tables):
        for entry in range(entry_size):
            
            # Add the Laplacian smoothing to the number of occurrences
            prob_table[tab][entry] += lap_smooth

            # Divide each entry of each table by the total number of number of occurrences of the given label
            # plus some Laplacian smoothing factor
            prob_table[tab][entry] /= ((prior_est[int(tab / num_unique_marks)] * num_labels) + lap_smooth)

            # Storing the log of the probabilities to avoid underflow. If there is at least one occurence, of
            # of a feature for a given label, store the log of the occurrence. If there is no occurence,
            # (so the value == 0), then give it some minimum value. (log(0) == -INF)
            if prob_table[tab][entry] > 0:
                prob_table[tab][entry] = math.log(prob_table[tab][entry])
            else:
                prob_table[tab][entry] = -20

#    print prob_table[2]
#    print prior_est

#    for i in range(30):
#        print prob_table[i]
    return prob_table

def prediction(data, data_labels, prior_est, prob_table):
    
    entry_size = data[0].size
    num_data_labels = data_labels.size
#    num_unique_marks = len(np.unique(np.array(data[0])))
    num_unique_marks = 20
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
#               print data_lab, entry
#               print data_lab, entry, len(data), entry_size, data[0].size, data_labels.size
               tmp += prob_table[int(num_unique_marks * label) + int(data[data_lab, entry])][entry]
#               print tmp
            if tmp_prediction[0] < tmp:
                tmp_prediction[0] = tmp
                tmp_prediction[1] = label
        logprob_pred[data_lab] = tmp_prediction[0]
        prediction[data_lab] = tmp_prediction[1]

#    print logprob_pred
#    print prediction
    return logprob_pred, prediction

def validation(prediction, actual, if_digit):
    
    if len(prediction) != len(actual):
        print "The number of predictions do not match the actual number of entries"

    else:
        if if_digit == 1:
            num_unique = 10
        else:
            num_unique = 2

        conf_matrix = np.zeros((num_unique,num_unique))
        correct = 0
        
        for i in range(len(prediction)):
            conf_matrix[prediction[i]][actual[i]] += 1
            if prediction[i] == actual[i]:
                correct += 1

        accuracy = float(correct) / float(len(actual)) * 100

        return conf_matrix, accuracy


if __name__== "__main__":
    if len(sys.argv) < 5:
        print "Usage: python {0} datafile labelfile testfile testlabelfile".format(sys.argv[0])
        sys.exit(1)
   
    if sys.argv[1][:4] == "face":
        SOBEL_USE = 0
    else:
        SOBEL_USE = 1

    train_accuracy = []
    confusion_matrix = []
    for train_percent in range(10, 101, 10):
        print train_percent
        print
        tmp_acc = 0

        for j in range(10):
        #   Importing Training Data
#            print "Importing Training Data"
            # Gets the data, labels and classes of the data and label files
            data, labels, classes = dataset.opendata(sys.argv[1], sys.argv[2])
        
            data = map(lambda s: str2img(s), data)
            img_dim = (len(data[0]), len(data[0][0]))
            data = map(lambda s: np.reshape(s, s.shape[0] * s.shape[1], 1), data)
            data = np.matrix(data)
            labels = np.matrix(labels).T
        
        #   Extracting Training Data
        
            train_labels = []
            random.seed()
        
            for i in range(len(data)):
                if int(math.floor(100 * random.random())) <= train_percent:
                    train_labels.append(1)
                else:
                    train_labels.append(0)
            
            train_labels = np.array(train_labels)
            data = data[train_labels == 1]
            labels = labels[train_labels == 1]
        
        #   Applying Sobel Filter for Edge Detection
        
            if SOBEL_USE == 2:
#                print "Applying Sobel Filter"
                data = sobel_filter(data, img_dim)
            else:
                pass
#                print "Skipping Sobel Filter"
        
        #   Prior Distribution
#            print "Calculating Prior Distribution"
            param_est = labels_estimate(labels)
        
        #   Log Joint Probability Tables
#            print "Computing Log Joint Prob Tables"
            joint_tables = compute_jointprobtable(data, labels, param_est)
            
        #   Importing Test Data
#            print "Importing Test Data"
            testdata, testlabels, testclasses = dataset.opendata(sys.argv[3], sys.argv[4])
        
            testdata = map(lambda s: str2img(s), testdata)
            testdata = map(lambda s: np.reshape(s, s.shape[0] * s.shape[1], 1), testdata)
            testdata = np.matrix(testdata)
            testlabels = np.matrix(testlabels).T
        
        #   Apply Sobel Filter for Edge Detection
            if SOBEL_USE == 2:
#                print "Applying Sobel Filter"
                testdata = sobel_filter(testdata, img_dim)
            else:
                pass
#                print "Skipping Sobel Filter"
            
        #   Prediction
#            print "Making Predictions"
            logprob, predict = prediction(testdata, testlabels, param_est, joint_tables)
        
        #   Validation
#            print "Validating Predictions"
            confusion, accuracy = validation(predict, testlabels, SOBEL_USE)
            tmp_acc += accuracy


        train_accuracy.append(tmp_acc / 10.0)
        confusion_matrix.append(confusion)

    print confusion_matrix
    print train_accuracy
