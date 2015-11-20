"""
Description: Learning pybrain on postcode digits dataset
Author: Iva
Date: Oct 2015
Python version: 2.7.10 (venv2)
"""

import numpy as np
import time
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectPercentile, f_classif

####################################
# load data from from kaggle files - csv
np_train = np.genfromtxt('data/train.csv', delimiter=',', skip_header= True, dtype='uint8')
#np_test = np.genfromtxt('data/test.csv', delimiter=',', skip_header= True, dtype='uint8')


n=6
skf = StratifiedKFold(np_train[:,0].ravel(), n_folds=3, random_state=3476)
predictions=np.zeros_like(np_train[:,0])
for train_index, test_index in skf:
    print time.ctime()

    # a bit of feature selection
    fscore = SelectPercentile(f_classif, percentile = 50)
    Xtrain = np.copy(fscore.fit_transform(np_train[train_index, 1:], np_train[train_index, 0]))
    Xtest = np.copy(fscore.transform(np_train[test_index, 1:]))

    # define knn
    knn_clf = KNeighborsClassifier(n_neighbors=6, weights='distance', metric='cosine', algorithm='brute')
    knn_clf.fit(Xtrain, np_train[train_index, 0])

    print('here')
    # fitting - this takes long - cca 15min
    predictions[test_index] = knn_clf.predict(Xtest)
    accu = accuracy_score(np_train[test_index, 0], predictions[test_index])
    print("the accuracy of kNN is : %f" % accu)

accu = accuracy_score(np_train[:, 0], predictions)
print("the total valudation accuracy of kNN is : %f" % accu)