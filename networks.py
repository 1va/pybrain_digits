"""
Description: Networks source code for Learning pybrain on postcode digits dataset
Author: Iva
Date: 16. 10. 2015
Python version: 2.7.10 (venv2)
"""

"""
bellow defined nerworks for MNIST 28x28:
    net_full
    net_shared
    net shared2
    net_shared2_bias
"""

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.networks import FeedForwardNetwork
from pybrain.structure.modules import LinearLayer, SigmoidLayer, SoftmaxLayer
from pybrain.structure.moduleslice import ModuleSlice
from pybrain.structure.connections.shared import MotherConnection,SharedFullConnection, FullConnection

################################################
def net_full():
    return buildNetwork(28*28, 12, 10, outclass=SoftmaxLayer)

################################################
def net_shared(h1dim=8):
    net = FeedForwardNetwork()
    # make modules
    inp=LinearLayer(28*28,name='input')
    h1=SigmoidLayer(h1dim**2,name='hidden')
    outp=SoftmaxLayer(10,name='output')
    net.addInputModule(inp)
    net.addModule(h1)
    net.addOutputModule(outp)
    # make connections
    from shared_mesh import shared_mesh
    net = shared_mesh(net, inlayer=inp, outlayer=h1, k=5)
    net.addConnection(FullConnection(h1, outp))
    net.sortModules()
    return net

################################################
def net_shared2(h1dim=13,h2dim=5):
    net = FeedForwardNetwork()
    # make modules
    inp=LinearLayer(28*28,name='input')
    h1=SigmoidLayer(h1dim**2,name='hidden1st')
    h2=SigmoidLayer(h2dim**2,name='hidden2nd')
    outp=SoftmaxLayer(10,name='output')
    net.addInputModule(inp)
    net.addModule(h1)
    net.addModule(h2)
    net.addOutputModule(outp)
    # make connections
    from shared_mesh import shared_mesh
    net = shared_mesh(net, inlayer=inp, outlayer=h1, k=5, mc_name='mother1st')
    net = shared_mesh(net, inlayer=h1, outlayer=h2, k=5, mc_name='mother2nd')
    net.addConnection(FullConnection(h2, outp))
    net.sortModules()
    return net

n.addModule(BiasUnit(name='bias'))