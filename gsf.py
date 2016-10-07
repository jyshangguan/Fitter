from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt
from sedfit.fitter import mcmc
from optparse import OptionParser

#Parse the commands#
#-------------------#
parser = OptionParser()
parser.add_option("-f", "--file", dest="filename", default="input_emcee.py",
                  help="write report to FILE", metavar="FILE")
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")
(options, args) = parser.parse_args()

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
parTruth   = inputModule.parTruth
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
nuisance = ppDict["nuisance"]

print("#--------------------------------#")
print("emcee Info:")
for keys in emceeDict.keys():
    print("{0}: {1}".format(keys, emceeDict[keys]))
print("#--------------------------------#")
modelUnct = inputModule.modelUnct
#em = mcmc.EmceeModel(sedData, sedModel, modelUnct, imSampler)
em = mcmc.EmceeModel(sedData, sedModel, modelUnct)
p0 = [em.from_prior() for i in range(nwalkers)]
sampler = em.EnsembleSampler(nwalkers, threads=threads)

#Burn-in 1st
print( "\n{:*^35}".format(" {0}th iteration ".format(0)) )
em.run_mcmc(p0, iterations=iStep, printFrac=printFrac, thin=thin)
em.diagnose()
pmax = em.p_logl_max()
em.print_parameters(truths=parTruth, burnin=0)
em.plot_lnlike(filename="{0}_lnprob.png".format(targname), histtype="step")

#Burn-in rest iteration
for i in range(iteration-1):
    print( "\n{:*^35}".format(" {0}th iteration ".format(i+1)) )
    em.reset()
    ratio = ballR * ballT**i
    print("-- P1 ball radius ratio: {0:.3f}".format(ratio))
    p1 = em.p_ball(pmax, ratio=ratio)
    em.run_mcmc(p1, iterations=iStep, printFrac=printFrac)
    em.diagnose()
    pmax = em.p_logl_max()
    em.print_parameters(truths=parTruth, burnin=50)
    em.plot_lnlike(filename="{0}_lnprob.png".format(targname), histtype="step")

#Run MCMC
print( "\n{:*^35}".format(" Final Sampling ") )
em.reset()
ratio = ballR * ballT**i
print("-- P1 ball radius ratio: {0:.3f}".format(ratio))
p1 = em.p_ball(pmax, ratio=ratio)
em.run_mcmc(p1, iterations=rStep, printFrac=printFrac, thin=thin)
em.diagnose()
em.print_parameters(truths=parTruth, burnin=burnIn, low=psLow, center=psCenter, high=psHigh)
em.plot_lnlike(filename="{0}_lnprob.png".format(targname), histtype="step")

#Post process
em.Save_Samples("{0}_samples.txt".format(targname), burnin=burnIn, select=True, fraction=25)
em.plot_chain(filename="{0}_chain.png".format(targname), truths=parTruth)
em.plot_corner(filename="{0}_triangle.png".format(targname), burnin=burnIn,
               nuisance=nuisance, truths=parTruth,
               quantiles=[psLow/100., psCenter/100., psHigh/100.], show_titles=True,
               title_kwargs={"fontsize": 20})
#em.plot_fit(filename="{0}_result.png".format(targname), truths=parTruth, burnin=burnIn,
#            low=psLow, center=psCenter, high=psHigh)
fig, axarr = plt.subplots(2, 1)
fig.set_size_inches(10, 10)
em.plot_fit_spec(truths=parTruth, FigAx=(fig, axarr[0]), burnin=burnIn,
                 low=psLow, center=psCenter, high=psHigh)
em.plot_fit(truths=parTruth, FigAx=(fig, axarr[1]), burnin=burnIn,
            low=psLow, center=psCenter, high=psHigh)
plt.savefig("{0}_result.png".format(targname), bbox_inches="tight")
plt.close()
print("Post-processed!")
