"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import cPickle
#import pickle
import gzip

# Third-party libraries
import numpy as np
import bson

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

#training_data, validation_data, test_data = load_data_wrapper()

def insert_images(db_mnist):
    db_mnist.remove()
    tr_d, va_d, te_d = load_data()
    for i in range(len(tr_d[0])):
        image_array = np.asarray(np.asarray(tr_d[0][i].ravel())*255, dtype='uint8')
        image_byte = bson.binary.Binary(image_array.tostring())
        classification = str(tr_d[1][i])
        doc = {'name': 'img'+str(i),
               "image": image_byte,
               "class": classification}
        print(i)
        db_mnist.insert_one(doc)
    return None

#training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
#    training_results = [vectorized_result(y) for y in tr_d[1]]
#    training_data = zip(training_inputs, training_results)
#    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]

import pymongo
db_mnist = pymongo.MongoClient("192.168.0.99:30000")["google"]["mnist"]

#insert_images(db_mnist)

from PIL import Image
def download_images(mongo_collection, num=10):
    cursor = list(mongo_collection.find().limit(num))
    for i in range(len(cursor)):
        image_array = np.fromstring(cursor[i]['image'], dtype='uint8')
        img = Image.fromarray(image_array.reshape(28,28), 'L')
        file_name = 'tmp/Picture'+str(i)+'.png'
        img.save(file_name)
    return None

#download_images(db_mnist)

"""
import numpy as np
from pybrain.datasets import ClassificationDataSet

def build_dataset(mongo_collection, patch_size=28, orig_size=28, nb_classes=10):
    patch_size = min(patch_size, orig_size)
    trim = round((orig_size-patch_size)/2)
    ds = ClassificationDataSet(patch_size**2, target=1, nb_classes=nb_classes)
    cursor = list(mongo_collection.find())
    for one_image in cursor:
        # convert from binary to numpy array and transform
        img_array = np.fromstring(one_image["image"], dtype='uint8')
        img_crop = img_array.reshape(orig_size, orig_size)[trim:(trim+patch_size),trim:(trim+patch_size)]
        classification = float(one_image["class"])
        ds.addSample(img_crop.ravel(),classification)
    print('New dataset contains %d images.' % len(ds))
    return ds

import pymongo
db_mnist = pymongo.MongoClient("192.168.0.99:30000")["google"]["mnist"]
ds = build_dataset(db_mnist)
"""
