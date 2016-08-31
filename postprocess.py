#!/Users/jinyi/anaconda/bin/python

import sys
import importlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import corner
import matplotlib.pyplot as plt
from sedfit.fitter import basicclass as bc
from sedfit import model_functions as sedmf
from sedfit import fit_functions   as sedff
from sedfit.fitter import mcmc


#The code starts#
#################
print("################################")
print("# Galaxy SED Fitter postprocess#")
print("################################")
print("\n")

#Plot fitting results#
######################

#Load the input info#
#-------------------#
if len(sys.argv) > 1:
    moduleName = sys.argv[1]
else:
    moduleName = "input_emcee"
print("Input module: {0}".format(moduleName))
inputModule = importlib.import_module(moduleName)
targname = inputModule.targname

#Input SED data#
#--------------#
sedData = inputModule.sedData

#Build up the model#
#------------------#

parAddDict_all = {
    "DL": sedData.dl,
}
waveModel = inputModule.waveModel
funcLib = sedmf.funcLib
modelDict = inputModule.inputModelDict
sedModel = bc.Model_Generator(modelDict, funcLib, waveModel, parAddDict_all)
parAllList = sedModel.get_parVaryList()
parNumber  = len(parAllList)


#Fit with DNest4#
#---------------#

##Create the DNest4Model
mockDict = inputModule.mockDict
fTrue = mockDict.get("f_add", None)
logl_True = mockDict['logl_true']
if fTrue is None:
    modelUnct = False
else:
    modelUnct = True #Whether to consider the model uncertainty in the fitting
    parNumber += 1
    parAllList.append(np.log(fTrue))
print("True log likelihood: {0:.3f}".format(logl_True))
print("Calculated log likelihood: {0:.3f}".format(mcmc.log_likelihood(parAllList, sedData, sedModel)))


##Fitting results
if parNumber == 1:
    raise AssertionError("The parameter should be at least 2!")
ps = np.loadtxt("{0}_samples.txt".format(targname), delimiter=",")

#Plot the corner diagram
fig = corner.corner(ps)
plt.savefig("{0}_triangle.png".format(targname))
plt.close()

#Plot the SED data and fit
try:
    assert ps.shape[1] == parNumber
except:
    raise AssertionError("The posterior sample is problematic!")
##Calculate the optimized paramter values
parRangeList = map( lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                   zip(*np.percentile(ps, [16, 50, 84], axis=0, interpolation="linear")) )
parRangeList = np.array(parRangeList)
parBfList = parRangeList[:, 0] #The best fit parameters
uncUlList = parRangeList[:, 1] #The upperlimits of the parameter uncertainties
uncLlList = parRangeList[:, 2] #The lowerlimits of the parameter uncertainties
parUlList = parBfList + uncUlList #The upperlimits of the parameters
parLlList = parBfList - uncLlList #The lowerlimits of the parameters
sedModel.updateParList(parBfList)
yFitBf = sedModel.combineResult()
yFitBF_cmp = sedModel.componentResult()
sedModel.updateParList(parUlList)
yFitUl = sedModel.combineResult()
yFitUl_cmp = sedModel.componentResult()
sedModel.updateParList(parLlList)
yFitLl = sedModel.combineResult()
yFitLl_cmp = sedModel.componentResult()

### Plot the Data
cmpList = mockDict["components"]
parList = mockDict["parameters"]
xModel  = mockDict["x_model"]
loglTrue = mockDict["logl_true"]

if modelUnct:
    parList.append( np.log(fTrue) )
for loop in range(len(parBfList)):
    p = parBfList[loop]
    u = uncUlList[loop]
    l = uncLlList[loop]
    t = parAllList[loop]
    print("{0}+{1}-{2}, {3}".format(p, u, l, t))

fig = plt.figure()
cList = ["r", "g", "b", "m", "y", "c"]
plt.plot(waveModel, yFitBf, color="brown", linewidth=3.0)
plt.fill_between(waveModel, yFitLl, yFitUl, color="brown", alpha=0.1)
counter = 0
for modelName in sedModel._modelList:
    plt.plot(waveModel, yFitBF_cmp[modelName], color=cList[counter])
    plt.fill_between(waveModel, yFitLl_cmp[modelName], yFitUl_cmp[modelName],
                     color=cList[counter], alpha=0.1)
    counter += 1
ax = plt.gca()
#plt.errorbar(wave, flux, yerr=sigma, fmt=".k", markersize=10, elinewidth=2, capsize=2)
sedData.plot_sed(FigAx=(fig, ax))
yModel = np.zeros_like(xModel)
counter = 0
for y in cmpList:
    plt.plot(xModel, y, color=cList[counter], linestyle="--")
    yModel += y
    counter += 1
plt.plot(xModel, yModel, color="k", linewidth=1.5, linestyle="--")
ymax = np.max(yModel)
plt.ylim([1e-2, ymax*2.0])
plt.xscale("log")
plt.yscale("log")
plt.savefig("{0}_result.png".format(targname))
plt.close()
