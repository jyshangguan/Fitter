from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import sys
import types
import corner
import importlib
import numpy as np
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
#print(sedModel.get_parVaryRanges())


mockDict = inputModule.mockDict
fTrue = mockDict.get("f_add", None)
logl_True = mockDict['logl_true']
if fTrue is None:
    modelUnct = False
else:
    modelUnct = True #Whether to consider the model uncertainty in the fitting
    parAllList.append(np.log(fTrue))
print("True log likelihood: {0:.3f}".format(logl_True))
print("Calculated log likelihood: {0:.3f}".format(mcmc.log_likelihood(parAllList, sedData, sedModel)))

#Fit with MCMC#
#---------------#
emceeDict = inputModule.emceeDict
imSampler = emceeDict["sampler"]
nwalkers  = emceeDict["nwalkers"]
ntemps    = emceeDict["ntemps"]
burnIn    = emceeDict["burn-in"]
nSteps    = emceeDict["nsteps"]
thin      = emceeDict["thin"]
threads   = emceeDict["threads"]
printFrac = emceeDict["printfrac"]
ndim = len(parAllList)
print("#--------------------------------#")
print("emcee Info:")
for keys in emceeDict.keys():
    print("{0}: {1}".format(keys, emceeDict[keys]))
print("#--------------------------------#")
em = mcmc.EmceeModel(sedData, sedModel, modelUnct, imSampler)

if imSampler == "PTSampler":
    p0 = np.zeros((ntemps, nwalkers, ndim))
    for loop_t in range(ntemps):
        for loop_w in range(nwalkers):
            p0[loop_t, loop_w, :] = em.from_prior()
    sampler = em.PTSampler(ntemps, nwalkers, threads=threads)
elif imSampler == "EnsembleSampler":
    p0 = [em.from_prior() for i in range(nwalkers)]
    sampler = em.EnsembleSampler(nwalkers, threads=threads)
else:
    raise RuntimeError("Cannot recognise the sampler '{0}'!".format(imSampler))


"""
#Burn-in 1st
pos, lnprob, state = em.burn_in(p0, iterations=burnIn, printFrac=printFrac, thin=thin)
em.diagnose()
pmax = em.p_logl_max()
em.print_parameters(parAllList)

#Burn-in 2nd
sampler.reset()
p1 = em.p_ball(pmax, ratio=1e-1)
em.burn_in(p0, iterations=2*burnIn, printFrac=printFrac, thin=thin)
em.diagnose()
pmax = em.p_logl_max()
em.print_parameters(parAllList)
"""

#Run MCMC
#pos = em.p_ball(pmax, ratio=1e-1)
#em.run_mcmc(pos, iterations=nSteps, printFrac=printFrac, thin=thin)
em.run_mcmc(p0, iterations=burnIn+nSteps, printFrac=printFrac, thin=thin)
em.diagnose()
em.print_parameters(parAllList, burnin=burnIn)
print("Max log_likelihood: {0:.3e}".format(em.get_logl_max()))

#Post process
targname = inputModule.targname
em.plot_corner(filename="{0}_triangle.png".format(targname), truths=parAllList, burnin=burnIn)
em.plot_fit(filename="{0}_result.png".format(targname), truths=parAllList, burnin=burnIn)
em.Save_Samples("{0}_samples.txt".format(targname), burnin=burnIn)
print("Post-processed!")
