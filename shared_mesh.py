"""
Description: Special 2D connection for Learning pybrain on postcode digits dataset
Author: Iva
Date: Oct 2015
Python version: 2.7.10 (venv2)
"""
from pybrain.structure.connections.shared import MotherConnection,SharedFullConnection, FullConnection
from pybrain.structure.moduleslice import ModuleSlice


def shared_mesh(net, inlayer, outlayer, k=None, mc_name='mother', bias=True):
    """
    Creates full shared connections form 2D-patches (size kxk) in inlayer to each neuron in outlayer.
    Patches are placed in uniform 2D grid.
    If k not provided, it estimates k so that the patches overlap by 1/3
    If bias=true the function assume a module named 'bias' exists.
    """
    ink = int(inlayer.outdim**.5)
    outk = int(outlayer.indim**.5)
    if k==None:
        overlap = .34
        k = max(int(round(ink/(1-overlap)/outk)),1)

    k = min(ink,k)            #patch size
    grid = [int(round(x*(ink-k)/float(outk-1))) for x in range(outk)]  #starting point of each patch

    mc= [0 for x in range(k)]
    outSlice =  [[0 for j in grid] for i in grid]
    inSlice= [[[0 for j in grid] for i in grid] for x in range(k)]
    for x in range(k):
        mc[x] = MotherConnection(k,name=mc_name+str(x))      # reference to the connections for each ROW in the patch, becouse module slicing is easy for consequent indeces in pybrain
    for i in range(outk):      # row index
        for j in range(outk):  # column index
            out_neuron = i*outk+j
            outSlice[i][j] = ModuleSlice(outlayer,inSliceFrom=out_neuron,inSliceTo=out_neuron+1,outSliceFrom=out_neuron,outSliceTo=out_neuron+1)
            for x in range(k):
                in_neuron = (grid[i]+x)*ink+grid[j]     # first neuron in the specific row of wanted patch
                inSlice[x][i][j]=ModuleSlice(inlayer,inSliceFrom=in_neuron,inSliceTo=in_neuron+k,outSliceFrom=in_neuron,outSliceTo=in_neuron+k)
                net.addConnection(SharedFullConnection(mc[x],inSlice[x][i][j],outSlice[i][j]))
    if bias:
        net.addConnection(FullConnection(net['bias'],outlayer))
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
    net_test = shared_mesh(net_test, inp, outp, k=3, bias=False)
    net_test.sortModules()
    print net_test
    for i in net_test.connections.values()[0]:
        print ('%s (%d - %d) -> (%d - %d)' %(i.mother, i.inSliceFrom, i.inSliceTo -1 , i.outSliceFrom, i.outSliceTo-1))
