import sys
import types
import corner
import importlib
import numpy as np
import matplotlib.pyplot as plt
from sedfit.fitter import basicclass as bc
from sedfit import model_functions as sedmf
from sedfit import fit_functions   as sedff
from sedfit.fitter import mcmc

#The code starts#
#---------------#
print("############################")
print("# Galaxy SED Fitter starts #")
print("############################")
print("\n")

#Load the input info#
#-------------------#
if len(sys.argv) > 1:
    moduleName = sys.argv[1]
else:
    moduleName = "input_emcee"
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
waveModel = inputModule.waveModel
modelDict = inputModule.inputModelDict
sedModel = bc.Model_Generator(modelDict, funcLib, waveModel, parAddDict_all)
parAllList = sedModel.get_parVaryList()


#Fit with MCMC#
#---------------#

mockDict = inputModule.mockDict
fTrue = mockDict.get("f_add", None)
logl_True = mockDict['logl_true']
if fTrue is None:
    modelUnct = False
else:
    modelUnct = True #Whether to consider the model uncertainty in the fitting
    parAllList.append(np.log(fTrue))
em = mcmc.EmceeModel(sedData, sedModel, modelUnct)
print("True log likelihood: {0:.3f}".format(logl_True))
print("Calculated log likelihood: {0:.3f}".format(mcmc.log_likelihood(parAllList, sedData, sedModel)))

nwalkers = 100
p0 = [em.from_prior() for i in range(nwalkers)]
ndim = len(p0[0])
print("{0} walkers, {1} dimensions".format(nwalkers, ndim))
lnprob = mcmc.lnprob
sampler = em.EnsembleSampler(nwalkers, threads=4)

printFraction = 0.1
burnIn = 500
nSteps = 10000

print("MCMC is burning-in...")
for i, (pos, lnprob, state) in enumerate(sampler.sample(p0, iterations=burnIn)):
    if not i % int(printFraction * burnIn):
        print("{0}%".format(100. * i / burnIn))
print("Burn-in finishes!")

sampler.reset()
print("MCMC is running...")
for i, (pos, lnprob, state) in enumerate(sampler.sample(pos, iterations=nSteps)):
    if not i % int(printFraction * nSteps):
        print("{0}%".format(100. * i / nSteps))
print("MCMC finishes!")

samples = sampler.chain.reshape((-1, ndim))
fig = corner.corner(samples)
plt.show()
