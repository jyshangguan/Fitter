#!/Users/jinyi/anaconda/bin/python

from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib_version = eval(matplotlib.__version__.split(".")[0])
if matplotlib_version > 1:
    plt.style.use("classic")
plt.rc('font',family='Times New Roman')
import sys
import types
import numpy as np
import cPickle as pickle
import sedfit.SED_Toolkit as sedt
from sedfit.fitter import basicclass as bc
from sedfit.mcmc import mcmc_emcee as mcmc
from sedfit import sedclass as sedsc
from sedfit import model_functions as sedmf

#Parse the commands#
#-------------------#
fitrsFile = sys.argv[1]
fp = open(fitrsFile, "r")
fitrs = pickle.load(fp)
fp.close()

#The code starts#
#################
print("#################################")
print("# Galaxy SED Fitter postprocess #")
print("#################################")

################################################################################
#                                    Data                                      #
################################################################################
dataPck = fitrs["dataPck"]
targname = dataPck["targname"]
redshift = dataPck["redshift"]
distance = dataPck["distance"]
dataDict = dataPck["dataDict"]
sedPck = dataPck["sedPck"]
sedData = sedsc.setSedData(targname, redshift, distance, dataDict, sedPck, silent=True)


################################################################################
#                                   Model                                      #
################################################################################
modelPck = fitrs["modelPck"]
modelDict = modelPck["modelDict"]
print("The model info:")
parCounter = 0
for modelName in modelDict.keys():
    print("[{0}]".format(modelName))
    model = modelDict[modelName]
    for parName in model.keys():
        param = model[parName]
        if not isinstance(param, types.DictType):
            continue
        elif param["vary"]:
            print("-- {0}, {1}".format(parName, param["type"]))
            parCounter += 1
        else:
            pass
print("Varying parameter number: {0}".format(parCounter))
print("#--------------------------------#")

#Build up the model#
#------------------#
parAddDict_all = modelPck["parAddDict_all"]
funcLib    = sedmf.funcLib
waveModel = modelPck["waveModel"]
sedModel = bc.Model_Generator(modelDict, funcLib, waveModel, parAddDict_all)
parTruth = modelPck["parTruth"]   #Whether to provide the truth of the model
modelUnct = modelPck["modelUnct"] #Whether to consider the model uncertainty in the fitting
parAllList = sedModel.get_parVaryList()
if modelUnct:
    parAllList.append(-np.inf)
    parAllList.append(-np.inf)
    parAllList.append(-5)

#Build the emcee object#
#----------------------#
em = mcmc.EmceeModel(sedData, sedModel, modelUnct)

#posterior process settings#
#--------------------------#
ppDict   = fitrs["ppDict"]
psLow    = ppDict["low"]
psCenter = ppDict["center"]
psHigh   = ppDict["high"]
nuisance = ppDict["nuisance"]
fraction = 0
burnIn = 0
ps = fitrs["posterior_sample"]

#Plot the SED data and fit
sedwave = sedData.get_List("x")
sedflux = sedData.get_List("y")
xmin = np.min(sedwave) * 0.9
xmax = np.max(sedwave) * 1.1
xlim = [xmin, xmax]
ymin = np.min(sedflux) * 0.5
ymax = np.max(sedflux) * 2.0
ylim = [ymin, ymax]
if sedData.check_csData():
    fig, axarr = plt.subplots(2, 1)
    fig.set_size_inches(10, 10)
    em.plot_fit_spec(truths=parTruth, FigAx=(fig, axarr[0]), nSamples=100,
                     burnin=burnIn, fraction=fraction, ps=ps)
    em.plot_fit(truths=parTruth, FigAx=(fig, axarr[1]), xlim=xlim, ylim=ylim,
                nSamples=100, burnin=burnIn, fraction=fraction, ps=ps)
    axarr[0].set_xlabel("")
    axarr[0].set_ylabel("")
    axarr[0].text(0.05, 0.8, targname,
                  verticalalignment='bottom', horizontalalignment='left',
                  transform=axarr[0].transAxes, fontsize=24,
                  bbox=dict(facecolor='white', alpha=0.5, edgecolor="none"))
else:
    fig = plt.figure(figsize=(7, 7))
    ax = plt.gca()
    em.plot_fit(truths=parTruth, FigAx=(fig, ax), xlim=xlim, ylim=ylim, nSamples=100,
                burnin=burnIn, fraction=fraction, ps=ps)
plt.savefig("{0}_result.png".format(targname), bbox_inches="tight")
plt.close()
print("Best fit plot finished!")

"""
#Plot the corner diagram
em.plot_corner(filename="{0}_triangle.png".format(targname), burnin=burnIn, ps=ps,
               nuisance=nuisance, truths=parTruth,  fraction=fraction,
               quantiles=[psLow/100., psCenter/100., psHigh/100.], show_titles=True,
               title_kwargs={"fontsize": 20})
print("Triangle plot finished!")
"""
