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
"""
modelDict = gaussModel.get_modelParDict()
for modelName in gaussModel._modelList:
    model = modelDict[modelName]
    for parName in model.keys():
        parDict = model[parName]
        print("{0}: {1} ({2[0]}, {2[1]})".format(parName, parDict["value"], parDict["range"]))
"""

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
gen = sampler.sample(max_num_levels=30, num_steps=1000, new_level_interval=10000,
                      num_per_step=10000, thread_steps=100,
                      num_particles=5, lam=10, beta=100, seed=1234)

# Do the sampling (one iteration here = one particle save)
for i, sample in enumerate(gen):
    print("# Saved {k} particles.".format(k=(i+1)))

# Run the postprocessing
dnest4.postprocess()

"""
#Plot the result#
#---------------#

ps = np.loadtxt("posterior_sample.txt")
nGauss = len(pValue)
for loop in range(nGauss):
    print "{0[0]}, {0[1]}, {0[2]}".format(pValue[loop])

#Calculate the optimized paramter values
parRangeList = map( lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                   zip(*np.percentile(ps, [16, 50, 84], axis=0)) )

par50List= []
par16List= []
par84List= []
print("Fitting results:")
for loop in range(nGauss):
    na = loop * 3 + 0
    nb = loop * 3 + 1
    nc = loop * 3 + 2
    prA = parRangeList[na]
    prB = parRangeList[nb]
    prC = parRangeList[nc]
    par50List.append( (prA[0], prB[0], prC[0]) )
    par84List.append( (prA[1], prB[1], prC[1]) )
    par16List.append( (prA[2], prB[2], prC[2]) )
    a_true, b_true, c_true = pValue[loop]
    print( "a_{0}: {1[0]}+{1[1]}-{1[2]} (True: {2})".format(loop, prA, a_true) )
    print( "b_{0}: {1[0]}+{1[1]}-{1[2]} (True: {2})".format(loop, prB, b_true) )
    print( "c_{0}: {1[0]}+{1[1]}-{1[2]} (True: {2})".format(loop, prC, c_true) )
print("-----------------")

fig = plt.figure()
plt.errorbar(xData, yObsr, yerr=yErr, fmt=".k")
plt.plot(xData, yTrue, linewidth=1.5, color="k")
for y in cmpList:
    plt.plot(xData, y, linestyle='--')

xm = np.linspace(1., 1000., 1000)
ym = np.zeros_like(xm)
for loop in range(nGauss):
    a50, b50, c50 = par50List[loop]
    y50 = GaussFunc(a50, b50, c50, xm)
    ym += y50
    a84, b84, c84 = par84List[loop]
    y84 = GaussFunc(a50+a84, b50+b84, c50+c84, xm)
    a16, b16, c16 = par16List[loop]
    y16 = GaussFunc(a50-a16, b50-b16, c50-c16, xm)
    plt.plot(xm, y50, color="r")
    plt.fill_between(xm, y16, y84, color="r", alpha=0.3)
plt.plot(xm, ym, color="r")
plt.show()

"""

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
