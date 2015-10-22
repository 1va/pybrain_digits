"""
Description: Special 2D connection for Learning pybrain on postcode digits dataset
Author: Iva
Date: 21. 10. 2015
Python version: 2.7.10 (venv2)
"""
from pybrain.structure.connections.shared import MotherConnection,SharedFullConnection
from pybrain.structure.moduleslice import ModuleSlice


def shared_mesh(net, inlayer, outlayer, k=None, mc_name='mother', bias=True):
    """
    Creates full shared connections form 2D-patches (size kxk) in inlayer to each neuron in outlayer.
    Patches are placed in uniform 2D grid.
    If k not provided, it estimates k so that the patches overlap by 1/3
    """
    ink = int(inlayer.outdim**.5)
    outk = int(outlayer.indim**.5)
    if k==None:
        overlap = .34
        k = max(int(round(ink/(1-overlap)/outk)),1)

    k = min(ink,k)            #patch size
    grid = [int(round(x*(ink-k)/float(outk-1))) for x in range(outk)]  #starting point of each patch

    mc= [0 for x in range(k)]
    outSlice =  [[0 for i in grid] for j in grid]
    inSlice= [[[0 for i in grid] for j in grid] for x in range(k)]
    for x in range(k):
        mc[x] = MotherConnection(k,name=mc_name+str(x))
    for i in range(outk):
        for j in range(outk):
            out_neuron = i*outk+j
            outSlice[i][j] = ModuleSlice(outlayer,inSliceFrom=out_neuron,inSliceTo=out_neuron+1,outSliceFrom=out_neuron,outSliceTo=out_neuron+1)
            for x in range(k):
                in_neuron = (grid[i]+x)*ink+grid[j]
                inSlice[x][i][j]=ModuleSlice(inlayer,inSliceFrom=in_neuron,inSliceTo=in_neuron+k,outSliceFrom=in_neuron,outSliceTo=in_neuron+k)
                net.addConnection(SharedFullConnection(mc[x],inSlice[x][i][j],outSlice[i][j]))
    if bias:
        pass
    return net

##############################
#build a small test network
if False:
    from pybrain.structure.networks import FeedForwardNetwork
    from pybrain.structure.modules import LinearLayer, SigmoidLayer, SoftmaxLayer
    net_test = FeedForwardNetwork()
    # make modules
    inp=LinearLayer(5*5,name='input')
    outp=SigmoidLayer(2*2,name='output')
    net_test.addInputModule(inp)
    net_test.addOutputModule(outp)
    shared_mesh(net_test, inp, outp, k=3)
    net_test.sortModules()
    print net_test

