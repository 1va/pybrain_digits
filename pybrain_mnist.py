"""
Description: Learning pybrain on postcode digits dataset
Author: Iva
Date: 16. 10. 2015
Python version: 2.7.10 (venv2)
"""

import numpy as np
import time
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cross_validation import train_test_split

from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError

####################################
# load data from from kaggle files - csv
np_train = np.genfromtxt('data/train.csv', delimiter=',', skip_header= True, dtype='uint8')
np_test = np.genfromtxt('data/test.csv', delimiter=',', skip_header= True, dtype='uint8')

dset = ClassificationDataSet(np_train.shape[1] - 1, 1)
dset.setField('input', np_train[:,1:])
dset.setField('target', np_train[:,:1])
dset._convertToOneOfMany( )
labels = dset['class'].ravel().tolist()

##############################################

from networks import net_shared2
net=net_shared2()

trainer = BackpropTrainer(net, dset, verbose=True)
for i in range(7):
    start = time.time()
    err = trainer.train()
    if i%3==0:
        out = trainer.testOnClassData()
        accu = accuracy_score(out,labels)
        print('Error after %d. iteration is %f, accuracy is %f. Last epoch lasted %f min.'
                %(i+1, err, accu, (time.time()-start)/60))

################################################

print out[0:40]
print labels[0:40]
print confusion_matrix(labels, out)
print accuracy_score(labels, out)
print percentError(out, dset['class'])

###################################################

from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

NetworkWriter.writeToFile(net, 'net_shared_13-5.xml')
#net = NetworkReader.readFrom('filename.xml')