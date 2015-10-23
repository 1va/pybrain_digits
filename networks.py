"""
Description: Networks source code for Learning pybrain on postcode digits dataset
Author: Iva
Date: Oct 2015
Python version: 2.7.10 (venv2)
"""

"""
bellow defined nerworks for MNIST 28x28:
    net_full
    net_shared
    net shared2
"""

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.networks import FeedForwardNetwork
from pybrain.structure.modules import LinearLayer, SigmoidLayer, SoftmaxLayer, BiasUnit
from pybrain.structure.moduleslice import ModuleSlice
from pybrain.structure.connections.shared import MotherConnection,SharedFullConnection, FullConnection
from shared_mesh import shared_mesh

################################################
def net_full(bias=True):
    return buildNetwork(28*28, 12, 10, outclass=SoftmaxLayer, bias=bias)

################################################
def net_shared(h1dim=8, bias=True):
    net = FeedForwardNetwork()
    # make modules
    inp=LinearLayer(28*28,name='input')
    h1=SigmoidLayer(h1dim**2,name='hidden')
    outp=SoftmaxLayer(10,name='output')
    net.addInputModule(inp)
    net.addModule(h1)
    net.addOutputModule(outp)
    if bias:
        net.addModule(BiasUnit(name='bias'))
        net.addConnection(FullConnection(net['bias'],outp))
    # make connections
    net = shared_mesh(net, inlayer=inp, outlayer=h1, k=5, bias=bias)
    net.addConnection(FullConnection(h1, outp))
    net.sortModules()
    return net

################################################
def net_shared2(h1dim=13,h2dim=5, bias=True):  #  alternative default: h1dim=8, h2dim=4,
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
    if bias:
        net.addModule(BiasUnit(name='bias'))
        net.addConnection(FullConnection(net['bias'],outp))
    # make connections
    net = shared_mesh(net, inlayer=inp, outlayer=h1, k=5, mc_name='mother1st', bias=bias)
    net = shared_mesh(net, inlayer=h1, outlayer=h2, k=5, mc_name='mother2nd', bias=bias)
    net.addConnection(FullConnection(h2, outp))
    net.sortModules()
    return net