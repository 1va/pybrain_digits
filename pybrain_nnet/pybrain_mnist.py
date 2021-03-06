"""
Description: Learning pybrain on postcode digits dataset
Author: Iva
Date: Oct 2015
Python version: 2.7.10 (venv2)
"""

import time

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import cross_validation              #train_test_split, cross_validation.StratifiedKFold, StratifiedShuffleSplit

from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain_nnet.networks import *
from pybrain.tools.xml.networkwriter import NetworkWriter


################################################
from pybrain.tools.xml.networkreader import NetworkReader
#NetworkWriter.writeToFile(net, 'net_shared_13-5.xml')
#net = NetworkReader.readFrom('filename.xml')

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

net=net_shared2()

trainer = BackpropTrainer(net,trainset)
for i in range(10):
    print (time.ctime() + ': Training epoch ' + str(i+1) + ' started')
    err = trainer.train()
    if i%1==0:
        out = trainer.testOnClassData()
        accu = accuracy_score(out,trainlabels)
        print('%s Error after %d. iteration is %f, accuracy is %f.'
                %(time.ctime(), i+1, err, accu))

################################################

print out[0:40]
print labels[0:40]
print confusion_matrix(labels, out)
print accuracy_score(labels, out)
print percentError(out, dset['class'])

###################################################
# using validation set to look for overfitting and optimize epoch num

def big_training(np_data, num_nets=1, num_epoch=20, net_builder=net_full, train_size=.1, testing=False):
    sss = cross_validation.StratifiedShuffleSplit(np_data[:,:1].ravel(), n_iter=num_nets , test_size=1-train_size, random_state=3476)
    nets=[None for net_ind in range(num_nets)]
    trainaccu=[[0 for i in range(num_epoch)] for net_ind in range(num_nets)]
    testaccu=[[0 for i in range(num_epoch)] for net_ind in range(num_nets)]
    net_ind=0
    for train_index, test_index in sss:
        print ('%s Building %d. network.' %(time.ctime(), net_ind+1))
        #print("TRAIN:", len(train_index), "TEST:", len(test_index))
        trainset = ClassificationDataSet(np_data.shape[1] - 1, 1)
        trainset.setField('input', np_data[train_index,1:]/100-.6)
        trainset.setField('target', np_data[train_index,:1])
        trainset._convertToOneOfMany( )
        trainlabels = trainset['class'].ravel().tolist()
        if testing:
            testset = ClassificationDataSet(np_data.shape[1] - 1, 1)
            testset.setField('input', np_data[test_index,1:]/100-.6)
            testset.setField('target', np_data[test_index,:1])
            testset._convertToOneOfMany( )
            testlabels = testset['class'].ravel().tolist()
        nets[net_ind] = net_builder()
        trainer = BackpropTrainer(nets[net_ind], trainset)
        for i in range(num_epoch):
            for ii in range(3):
                err = trainer.train()
            print ('%s Epoch %d: Network trained with error %f.' %(time.ctime(), i+1, err))
            trainaccu[net_ind][i]=accuracy_score(trainlabels,trainer.testOnClassData())
            print ('%s Epoch %d: Train accuracy is %f' %(time.ctime(), i+1, trainaccu[net_ind][i]))
            print ([sum([trainaccu[y][i]>tres for y in range(net_ind+1)]) for tres in [0,.1,.2,.3,.4,.5,.6]])
            if testing:
                testaccu[net_ind][i]=accuracy_score(testlabels,trainer.testOnClassData(testset))
                print ('%s Epoch %d: Test accuracy is %f' %(time.ctime(), i+1, testaccu[net_ind][i]))
        NetworkWriter.writeToFile(nets[net_ind], 'nets/'+net_builder.__name__+str(net_ind)+'.xml')
        net_ind +=1
    return [nets, trainaccu, testaccu]

result_full = big_training(np_train[1:5000,:], num_nets=3, num_epoch=7, net_builder=net_full)

result_shared2 = big_training(np_train[20000:40000,:], num_nets=5, num_epoch=7, net_builder=net_shared2)

result_shared = big_training(np_train[1:20000,:], num_nets=3, num_epoch=20, net_builder=net_shared)
result_shared3 = big_training(np_train[20000:40000,:], num_nets=5, num_epoch=20, net_builder=net_shared3)
result_shared_multi = big_training(np_train[20000:40000,:], num_nets=5, num_epoch=20, net_builder=net_shared_multi)

result_shared_multi = big_training(np_train, num_nets=100, num_epoch=1, train_size=.01, net_builder=net_shared_multi)


testset = ClassificationDataSet(np_train.shape[1] - 1, 1)
testset.setField('input', np_train[20000:30000,1:])
testset.setField('target', np_train[20000:30000,:1])
testset._convertToOneOfMany( )
testlabels = testset['class'].ravel().tolist()

out=[0,0,0]
for i in range(3):
    net = NetworkReader.readFrom('nets/net_shared'+str(i)+'.xml')
    trainer = BackpropTrainer(net)
    out[i] = trainer.testOnClassData(testset)

out[3]=testlabels
for i in range(4):
    for j in range(i+1,4):
        print ('Accuracy between %d and %d is %f.' %(i, j, accuracy_score(out[i],out[j]) ))

from scipy.stats import mode
m=np.zeros((len(testlabels),2))
for i in range(len(testlabels)):
    x=mode([out[0][i], out[1][i], out[2][i]])
    m[i,:]=x.mode.tolist()+x.count.tolist()


###################
