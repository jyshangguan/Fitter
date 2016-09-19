from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import sys
import types
import corner
import importlib
import numpy as np
from sedfit.fitter import mcmc
from optparse import OptionParser
from emcee.utils import MPIPool

#Parse the commands#
#-------------------#
parser = OptionParser()
parser.add_option("-f", "--file", dest="filename", default="input_emcee.py",
                  help="write report to FILE", metavar="FILE")
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")
parser.add_option("-m", "--mpi",
                  action="store_true", dest="runmpi", default=False,
                  help="run the code with multiple cores")
(options, args) = parser.parse_args()

#Parallel computing setup#
#------------------------#
runMPI = options.runmpi
if runMPI:
    # Initialize the MPI-based pool used for parallelization.
    pool = MPIPool()
    if not pool.is_master():
        # Wait for instructions from the master process.
        pool.wait()
        sys.exit(0)
    print("**Running with MPI...")
else:
    print("**Running with multiple threads...")

#The code starts#
#---------------#
print("############################")
print("# Galaxy SED Fitter starts #")
print("############################")

#Load the input module#
#---------------------#
moduleName = options.filename
print("Input module: {0}".format(moduleName))
inputModule = importlib.import_module(moduleName.split(".")[0])
targname = inputModule.targname

#Input SED data#
#--------------#
sedData = inputModule.sedData

#Input model#
#-----------#
sedModel = inputModule.sedModel
parAllList = inputModule.parAllList
ndim = len(parAllList)


#Fit with MCMC#
#---------------#
emceeDict = inputModule.emceeDict
imSampler = emceeDict["sampler"]
nwalkers  = emceeDict["nwalkers"]
ntemps    = emceeDict["ntemps"]
iteration = emceeDict["iteration"]
iStep     = emceeDict["iter-step"]
ballR     = emceeDict["ball-r"]
ballT     = emceeDict["ball-t"]
rStep     = emceeDict["run-step"]
burnIn    = emceeDict["burn-in"]
thin      = emceeDict["thin"]
threads   = emceeDict["threads"]
printFrac = emceeDict["printfrac"]
ppDict   = inputModule.ppDict
psLow    = ppDict["low"]
psCenter = ppDict["center"]
psHigh   = ppDict["high"]

print("#--------------------------------#")
print("emcee Info:")
for keys in emceeDict.keys():
    print("{0}: {1}".format(keys, emceeDict[keys]))
print("#--------------------------------#")
modelUnct = inputModule.modelUnct
em = mcmc.EmceeModel(sedData, sedModel, modelUnct, imSampler)

if imSampler == "PTSampler":
    p0 = np.zeros((ntemps, nwalkers, ndim))
    for loop_t in range(ntemps):
        for loop_w in range(nwalkers):
            p0[loop_t, loop_w, :] = em.from_prior()
    sampler = em.PTSampler(ntemps, nwalkers, threads=threads)
elif imSampler == "EnsembleSampler":
    p0 = [em.from_prior() for i in range(nwalkers)]
    sampler = em.EnsembleSampler(nwalkers, threads=threads, a=2.0)
else:
    raise RuntimeError("Cannot recognise the sampler '{0}'!".format(imSampler))

#Burn-in 1st
print( "\n{:*^35}".format(" {0}th iteration ".format(0)) )
em.run_mcmc(p0, iterations=iStep, printFrac=printFrac, thin=thin)
em.diagnose()
pmax = em.p_logl_max()
em.print_parameters(parAllList, burnin=50)
em.plot_lnlike(filename="{0}_lnprob.png".format(targname), histtype="step")

#Burn-in rest iteration
for i in range(iteration-1):
    print( "\n{:*^35}".format(" {0}th iteration ".format(i+1)) )
    em.reset()
    ratio = ballR * ballT**i
    print("-- P1 ball radius ratio: {0:.3f}".format(ratio))
    p1 = em.p_ball(pmax, ratio=ratio)
    em.run_mcmc(p1, iterations=iStep, printFrac=printFrac, thin=thin)
    em.diagnose()
    pmax = em.p_logl_max()
    em.print_parameters(parAllList, burnin=50)
    em.plot_lnlike(filename="{0}_lnprob.png".format(targname), histtype="step")

#Run MCMC
print( "\n{:*^35}".format(" Final Sampling ") )
em.reset()
ratio = ballR * ballT**i
print("-- P1 ball radius ratio: {0:.3f}".format(ratio))
p1 = em.p_ball(pmax, ratio=ratio)
em.run_mcmc(p1, iterations=rStep, printFrac=printFrac, thin=thin)
em.diagnose()
em.print_parameters(parAllList, burnin=burnIn, low=psLow, center=psCenter, high=psHigh)
em.plot_lnlike(filename="{0}_lnprob.png".format(targname), histtype="step")

#Close the pools
if runMPI:
    pool.close()

#Post process
em.plot_chain(filename="{0}_chain.png".format(targname), truths=parAllList)
em.plot_corner(filename="{0}_triangle.png".format(targname), burnin=burnIn, truths=parAllList,
               quantiles=[psLow/100., psCenter/100., psHigh/100.], show_titles=True,
               title_kwargs={"fontsize": 20})
em.plot_fit(filename="{0}_result.png".format(targname), truths=parAllList, burnin=burnIn,
            low=psLow, center=psCenter, high=psHigh)
em.Save_Samples("{0}_samples.txt".format(targname), burnin=0)
print("Post-processed!")
