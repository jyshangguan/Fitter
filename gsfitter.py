#!/Users/jinyi/anaconda/bin/python

import sys
import importlib
import dnest4
import numpy as np
import matplotlib.pyplot as plt
from sedfit.fitter import basicclass as bc
from sedfit import model_functions as sedmf
from sedfit import fit_functions   as sedff

#The code starts#
#---------------#
print("############################")
print("# Galaxy SED Fitter starts #")
print("############################")
print("\n")

#Load the input info#
#-------------------#
moduleName = sys.argv[1]
print("Input module: {0}".format(moduleName))
inputModule = importlib.import_module(moduleName)

#Input SED data#
#--------------#
sedData = inputModule.sedData

#Build up the model#
#------------------#

parAddDict_all = {
    "DL": sedData.dl,
}
funcLib    = sedmf.funcLib
Model2Data = sedmf.Model2Data
waveModel = inputModule.waveModel
modelDict = inputModule.inputModelDict
sedModel = bc.Model_Generator(modelDict, funcLib, waveModel, parAddDict_all,
                              model2data=Model2Data)
parAllList = sedModel.get_parVaryList()


#Fit with DNest4#
#---------------#

## Create the DNest4Model
mockDict = inputModule.mockDict
fTrue = mockDict.get("f_add", None)
logl_True = mockDict['logl_true']
if fTrue is None:
    modelUnct = False
else:
    modelUnct = True #Whether to consider the model uncertainty in the fitting
    parAllList.append(np.log(fTrue))
logLFunc = sedff.logLFunc_SED
dn4m = bc.DNest4Model(sedData, sedModel, logLFunc, modelUnct)
print("True log likelihood: {0:.3f}".format(logl_True))
print("Calculated log likelihood: {0:.3f}".format(dn4m.log_likelihood(parAllList)))
#dn4m.perturb(parAllList)
#print dn4m.log_likelihood(parAllList)

"""
fig, ax = sedData.plot_sed()
fig, ax = sedModel.plot(FigAx=(fig, ax))
plt.ylim([1e-2, 1e4])
plt.show()
#"""

## Create a model object and a sampler
sampler = dnest4.DNest4Sampler(dn4m,
                               backend=dnest4.backends.CSVBackend(".",
                                                                  sep=" "))

## Set up the sampler. The first argument is max_num_levels
DN4options = inputModule.DN4options
gen = sampler.sample(max_num_levels    = DN4options["max_num_levels"],
                     num_steps         = DN4options["num_steps"],
                     new_level_interval= DN4options["new_level_interval"],
                     num_per_step      = DN4options["num_per_step"],
                     thread_steps      = DN4options["thread_steps"],
                     num_particles     = DN4options["num_particles"],
                     lam               = DN4options["lam"],
                     beta              = DN4options["beta"],
                     seed              = DN4options["seed"])

## Do the sampling (one iteration here = one particle save)
for i, sample in enumerate(gen):
    print("# Saved {k} particles.".format(k=(i+1)))
