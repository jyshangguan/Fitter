import numpy as np
import dnest4
from dnest4.utils import rng
import fitter.basicclass as bc
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


#Load the data#
#-------------#
Nmodel = 5
dataName = "gauss{0}".format(Nmodel)
try:
    fp = open("{0}.dict".format(dataName), "r")
except IOError:
    raise IOError("{0} does not exist!".format(dataName))
model = pickle.load(fp)
fp.close()
xData = model["x"]
Ndata = len(xData)
fAdd  = model["f_add"]
yTrue = model['y_true']
yObsr = model['y_obsr']
yErr  = model['y_err']
pValue = model['parameters']
rangeList = model['ranges']
cmpList = model['compnents']

print( "#------Start fitting------#" )
print( "# Ndata: {0}".format(Ndata) )
print( "# Nmodel: {0}".format(Nmodel) )
print( "# f_add: {0}".format(fAdd) )
print( "#-------------------------#" )

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
modelDict = gaussModel.get_modelParDict()
for modelName in gaussModel._modelList:
    model = modelDict[modelName]
    for parName in model.keys():
        parDict = model[parName]
        print("{0}: {1} ({2})".format(parName, parDict["value"], parDict["range"]))

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
