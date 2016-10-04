#!/Users/jinyi/anaconda/bin/python

import sys
import importlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sedfit.fitter import mcmc
from optparse import OptionParser

#Parse the commands#
#-------------------#
parser = OptionParser()
parser.add_option("-f", "--file", dest="filename", default="input_emcee.py",
                  help="write report to FILE", metavar="FILE")
(options, args) = parser.parse_args()

#The code starts#
#################
print("#################################")
print("# Galaxy SED Fitter postprocess #")
print("#################################")

#Plot fitting results#
######################

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
em = mcmc.EmceeModel(sedData, sedModel, modelUnct, imSampler)

#Load posterior samplings
ps = np.loadtxt("{0}_samples.txt".format(targname), delimiter=",")

#Plot the corner diagram
em.plot_corner(filename="{0}_triangle.png".format(targname), burnin=burnIn,
               ps=ps, nuisance=nuisance, truths=parTruth,
               quantiles=[psLow/100., psCenter/100., psHigh/100.], show_titles=True,
               title_kwargs={"fontsize": 20})
print("Triangle plot finished!")

#Plot the SED data and fit
fig, axarr = plt.subplots(2, 1)
fig.set_size_inches(10, 10)
em.plot_fit_spec(truths=parTruth, FigAx=(fig, axarr[0]), burnin=burnIn,
                 low=psLow, center=psCenter, high=psHigh, ps=ps)
em.plot_fit(truths=parTruth, FigAx=(fig, axarr[1]), burnin=burnIn,
            low=psLow, center=psCenter, high=psHigh, ps=ps)
plt.savefig("{0}_result.png".format(targname), bbox_inches="tight")
plt.close()
print("Best fit plot finished!")
