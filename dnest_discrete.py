import numpy as np
import dnest4
from dnest4.utils import rng
import datafit_im.basicclass as bc
from gaussian_model import MultiGaussian, GaussianModelDiscrete
import matplotlib.pyplot as plt
import cPickle as pickle
import types

def logLFunc_simple(params, data, model):
        parDict = model.get_modelParDict()
        pIndex = 0
        for modelName in model._modelList:
            parFitDict = parDict[modelName]
            for parName in parFitDict.keys():
                parFitDict[parName]["value"] = params[pIndex]
                pIndex += 1
        x = np.array(data.get_List('x'))
        y = np.array(data.get_List('y'))
        e = np.array(data.get_List('e'))
        ym = np.array(model.combineResult(x))
        #Calculate the log_likelihood
        s2 = e**2.
        logL = -0.5 * np.sum( (y - ym)**2 / s2 + np.log(2 * np.pi * s2) )
        return logL


#Generate the data#
#-----------------#
Ndata  = 50
xMax   = 1000.0
Nmodel = 1
fAdd   = None #0.1
pRange = [
    [5.0, 20.0],   #The range of a
    [20.0, 580.0], #The range of b
    [10.0, 100.0], #The range of c
]
print( "#------Start fitting------#" )
print( "# Ndata: {0}".format(Ndata) )
print( "# Nmodel: {0}".format(Nmodel) )
print( "# f_add: {0}".format(fAdd) )
print( "#-------------------------#" )

xData = np.linspace(1.0, xMax, Ndata)
model = MultiGaussian(xData, pRange, Nmodel, fAdd)
yTrue = model['y_true']
yObsr = model['y_obsr']
yErr = model['y_err']
pValue = model['parameters']
rangeList = model['ranges']
cmpList = model['compnents']
model['x'] = xData
fp = open("test_model.dict", "w")
pickle.dump(model, fp)
fp.close()
print("test_model.dict is saved!")


#Generate the gaussian model#
#---------------------------#
fp = open("gt.dict", "r")
gt = pickle.load(fp)
fp.close()
pRangeDiscrete = {
    "a": list( np.arange(5.0, 20.0, 1.0) ),
    "b": list( np.arange(20.0, 580.0, 20.0) ),
    "c": list( np.arange(10.0, 100.0, 5.0) )
}
gaussModel = GaussianModelDiscrete(Nmodel, pRangeDiscrete, gt)
#print gaussModel.get_modelParDict()

"""
ym = gaussModel.combineResult(xData)
plt.plot(xData, ym)
plt.show()
"""

#Construct the DNest4Model#
#-------------------------#
nameList = ["p{0}".format(i) for i in range(Ndata)]
flagList = np.ones_like(xData)
dd = bc.DiscreteSet(nameList, xData, yObsr, yErr, flagList)
ddSet = {"gauss": dd}
ds = bc.DataSet(ddSet)
dn4m = bc.DNest4Model(ds, gaussModel, logLFunc_simple)
#print dn4m.log_likelihood([10., 320., 43.])


# Create a model object and a sampler
sampler = dnest4.DNest4Sampler(dn4m,
                               backend=dnest4.backends.CSVBackend(".",
                                                                  sep=" "))

# Set up the sampler. The first argument is max_num_levels
gen = sampler.sample(max_num_levels=30, num_steps=1000, new_level_interval=10000,
                      num_per_step=10000, thread_steps=100,
                      num_particles=5, lam=10, beta=100, seed=1234)

# Do the sampling (one iteration here = one particle save)
for i, sample in enumerate(gen):
    print("# Saved {k} particles.".format(k=(i+1)))

# Run the postprocessing
dnest4.postprocess()


#-----
#Tests
#-----
##Test the paramter type
"""
parDict = gaussModel.get_modelParDict()
parFitDict = parDict['G1']
parFitDict['a']['type'] = "d"
parFitDict['a']['range'] = range(1, 5)
print parFitDict
"""
#Plot the data
if False:
    plt.plot(xData, yTrue)
    plt.errorbar(xData, yObsr, yerr=yErr, fmt='.k')
    for y in cmpList:
        plt.plot(xData, y, linestyle='--')
    fig = plt.gcf()
    ax  = plt.gca()
    ylim = ax.get_ylim()
    gaussModel.plot(xData, FigAx=(fig, ax))
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_ylim(ylim)
    #plt.show()
#-------------------------
#Test the perturb function
#-------------------------
if False:
    print("Start test")
    p0 = np.array([10.0, 30.0, 20.0])
    pList = [p0.copy()]
    l0 = np.exp( dn4m.log_likelihood(p0) )
    print l0
    for loop in range(100000):
        p = p0.copy()
        h = np.exp( dn4m.perturb(p) )
        l = np.exp( dn4m.log_likelihood(p) )
        alpha = min([1, h * l/l0])
        r = rng.rand()
        #print h, l, alpha, r
        if r < alpha:
            pList.append(p)
            p0 = p
            l0 = l
    pList = np.array(pList)
    print pList
    plt.plot(pList[:, 0], pList[:, 1], marker='o', color='k')
    plt.plot(pList[0, 0], pList[0, 1], marker='o', color='r')
    plt.xlim([0, 30])
    plt.ylim([0, 600])
    plt.show()
    af = np.median(pList[:, 0])
    bf = np.median(pList[:, 1])
    cf = np.median(pList[:, 2])
    print "a: {0}, b: {1}, c: {2}".format(af, bf, cf)
    print pValue
