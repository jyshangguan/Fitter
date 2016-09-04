from __future__ import print_function
import sys
import types
import corner
import importlib
import numpy as np
from sedfit.fitter import basicclass as bc
from sedfit import model_functions as sedmf
from sedfit import fit_functions   as sedff
from sedfit.fitter import mcmc
from emcee.utils import MPIPool

# Initialize the MPI-based pool used for parallelization.
pool = MPIPool()

if not pool.is_master():
    # Wait for instructions from the master process.
    pool.wait()
    sys.exit(0)

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
iteration = emceeDict["iteration"]
ballR     = emceeDict["ball-r"]
ballT     = emceeDict["ball-t"]
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
em = mcmc.EmceeModel(sedData, sedModel, modelUnct)


#Burn-in 1st
sampler = em.EnsembleSampler(nwalkers, threads=threads)
p0 = np.array(em.p_prior())
print( "\n{:*^35}".format(" {0}th iteration ".format(0)) )
pos, lnprob, state = em.run_mcmc(p0, iterations=burnIn, printFrac=printFrac, thin=thin)
em.diagnose()
pmax = em.p_logl_max()
em.print_parameters(parAllList, burnin=50)

#Burn-in rest iteration
for i in range(iteration-1):
    print( "\n{:*^35}".format(" {0}th iteration ".format(i+1)) )
    em.reset()
    ratio = ballR * ballT**i
    print("-- P1 ball radius ratio: {0:.3f}".format(ratio))
    p1 = em.p_ball(pmax, ratio=ratio)
    em.run_mcmc(p1, iterations=burnIn, printFrac=printFrac, thin=thin)
    em.diagnose()
    pmax = em.p_logl_max()
    em.print_parameters(parAllList, burnin=50)

#Run MCMC
print( "\n{:*^35}".format(" Final Sampling ") )
sampler = em.PTSampler(ntemps, nwalkers, threads=threads)
em.reset()
ratio = ballR * ballT**i
print("-- P1 ball radius ratio: {0:.3f}".format(ratio))
p1 = em.p_ball(pmax, ratio=ratio)
em.run_mcmc(p1, iterations=nSteps, printFrac=printFrac, thin=thin)
em.diagnose()
em.print_parameters(parAllList, burnin=100, low=16, high=84)

#Close the pools
pool.close()

#Post process
targname = inputModule.targname
em.plot_corner(filename="{0}_triangle.png".format(targname), truths=parAllList, burnin=100)
em.plot_fit(filename="{0}_result.png".format(targname), truths=parAllList, burnin=100, low=16, high=84)
em.Save_Samples("{0}_samples.txt".format(targname), burnin=100)
print("Post-processed!")
