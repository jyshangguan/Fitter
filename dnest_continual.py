import os
import numpy as np
import dnest4
from dnest4.utils import rng
import fitter.basicclass as bc
from gaussian_model import MultiGaussian, GaussianModelSet, GaussFunc
import matplotlib.pyplot as plt
import cPickle as pickle
import types


#DNest4 model#
#------------#
class DNest4Model(object):
    """
    Specify the model
    """
    def __init__(self, data, model, logl):
        if isinstance(data, bc.DataSet):
            self.__data = data
        if isinstance(model, bc.ModelCombiner):
            self.__model = model
        if isinstance(logl, types.FunctionType):
            self._logl = logl

    def from_prior(self):
        """
        The prior of all the parameters are uniform.
        """
        parList = []
        parDict = self.__model.get_modelParDict()
        for modelName in self.__model._modelList:
            parFitDict = parDict[modelName]
            for parName in parFitDict.keys():
                parRange = parFitDict[parName]["range"]
                parType  = parFitDict[parName]["type"]
                if parType == "c":
                    #print "[DN4M]: continual"
                    r1, r2 = parRange
                    p = (r2 - r1) * rng.rand() + r1 #Uniform distribution
                elif parType == "d":
                    #print "[DN4M]: discrete"
                    p = np.random.choice(parRange, 1)[0]
                else:
                    raise TypeError("The parameter type '{0}' is not recognised!".format(parType))
                parList.append(p)
        parList = np.array(parList)
        return parList

    def perturb(self, params):
        """
        Each step we perturb all the parameters which is more effective from
        computation point of view.
        """
        logH = 0.0
        parDict = self.__model.get_modelParDict()
        pIndex = 0
        for modelName in self.__model._modelList:
            parFitDict = parDict[modelName]
            for parName in parFitDict.keys():
                parRange = parFitDict[parName]["range"]
                parType  = parFitDict[parName]["type"]
                if parType == "c":
                    #print "[DN4M]: continual"
                    r1, r2 = parRange
                    params[pIndex] += (r2 - r1) * dnest4.randh() #Uniform distribution
                    if (params[pIndex] < r1) or (params[pIndex] > r2):
                        #print "[DNest4Model]: perturb out boundary!"
                        logH -= np.inf
                elif parType == "d":
                    #print "[DN4M]: discrete"
                    rangeLen = len(parRange)
                    iBng = -1 * parRange.index(params[pIndex])
                    iPar = iBng + rng.randint(rangeLen)
                    params[pIndex] = parRange[iPar]
                    if not params[pIndex] in parRange:
                        #print "[DNest4Model]: perturb out boundary!"
                        logH -= np.inf
                else:
                    raise TypeError("The parameter type '{0}' is not recognised!".format(parType))
                parFitDict[parName]["value"] = params[pIndex]
                pIndex += 1
        return logH

    def log_likelihood(self, params):
        """
        Gaussian sampling distrubution.
        """
        logL  = self._logl(params, self.__data, self.__model)
        return logL

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
data = pickle.load(fp)
fp.close()
xData = data["x"]
Ndata = len(xData)
fAdd  = data["f_add"]
yTrue = data['y_true']
yObsr = data['y_obsr']
yErr  = data['y_err']
pValue = data['parameters']
rangeList = data['ranges']
cmpList = data['compnents']

print( "#------Start fitting------#" )
print( "# Ndata: {0}".format(Ndata) )
print( "# Nmodel: {0}".format(Nmodel) )
print( "# f_add: {0}".format(fAdd) )
print( "#-------------------------#" )

#Generate the gaussian model#
#---------------------------#
gaussModel = GaussianModelSet(pValue, rangeList)
modelDict = gaussModel.get_modelParDict()
for modelName in gaussModel._modelList:
    model = modelDict[modelName]
    for parName in model.keys():
        parDict = model[parName]
        print("{0}: {1} ({2[0]}, {2[1]})".format(parName, parDict["value"], parDict["range"]))

#Construct the DNest4Model#
#-------------------------#
nameList = ["p{0}".format(i) for i in range(Ndata)]
flagList = np.ones_like(xData)
dd = bc.DiscreteSet(nameList, xData, yObsr, yErr, flagList)
ddSet = {"gauss": dd}
ds = bc.DataSet(ddSet)
dn4m = DNest4Model(ds, gaussModel, logLFunc_simple)
#print dn4m.log_likelihood([10., 320., 43.])

# Create a model object and a sampler
sampler = dnest4.DNest4Sampler(dn4m,
                               backend=dnest4.backends.CSVBackend(".",
                                                                  sep=" "))

# Set up the sampler. The first argument is max_num_levels
gen = sampler.sample(max_num_levels=30, num_steps=2000, new_level_interval=10000,
                      num_per_step=10000, thread_steps=100,
                      num_particles=5, lam=10, beta=100, seed=1234)

# Do the sampling (one iteration here = one particle save)
for i, sample in enumerate(gen):
    print("# Saved {k} particles.".format(k=(i+1)))

# Run the postprocessing
dnest4.postprocess()

#Rename the posterior sample file name#
#-------------------------------------#
os.rename("posterior_sample.txt", "{0}_c_posterior.txt".format(dataName))
