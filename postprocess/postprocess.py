#!/Users/jinyi/anaconda/bin/python

from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import sys
import types
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import rel_SED_Toolkit as sedt
from sedfit.fitter import basicclass as bc
from sedfit.fitter import mcmc
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
sedPck = dataPck["sedPck"]
sed = sedPck["sed_cb"]
sedwave = sedPck["sed"][0]
sedflux = sedPck["sed"][1]
sedsigma = sedPck["sed"][2]
spcwave = sedPck["spc"][0]
spcflux = sedPck["spc"][1]
spcsigma = sedPck["spc"][2]
print("#--------------------------------#")
print("Target: {0}".format(targname))
print("Redshift: {0}".format(redshift))
print("#--------------------------------#")

## Put into the sedData
bandList = dataPck["bandList"]
sedName  = dataPck["sedName"]
spcName  = dataPck["spcName"]
if not sedName is None:
    sedflag = np.ones_like(sedwave)
    sedDataType = ["name", "wavelength", "flux", "error", "flag"]
    phtData = {sedName: bc.DiscreteSet(bandList, sedwave, sedflux, sedsigma, sedflag, sedDataType)}
else:
    phtData = {}
if not spcName is None:
    spcflag = np.ones_like(spcwave)
    spcDataType = ["wavelength", "flux", "error", "flag"]
    spcData = {"IRS": bc.ContinueSet(spcwave, spcflux, spcsigma, spcflag, spcDataType)}
else:
    spcData = {}
sedData = sedsc.SedClass(targname, redshift, phtDict=phtData, spcDict=spcData)
sedData.set_bandpass(bandList)


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
fraction = ppDict["fraction"]
ps = fitrs["posterior_sample"]
burnIn = 0

#Plot the corner diagram
em.plot_corner(filename="{0}_triangle.png".format(targname), burnin=burnIn, ps=ps,
               nuisance=nuisance, truths=parTruth,  select=True, fraction=fraction,
               quantiles=[psLow/100., psCenter/100., psHigh/100.], show_titles=True,
               title_kwargs={"fontsize": 20})
print("Triangle plot finished!")

#Plot the SED data and fit
fig, axarr = plt.subplots(2, 1)
fig.set_size_inches(10, 10)
em.plot_fit_spec(truths=parTruth, FigAx=(fig, axarr[0]),
                 burnin=burnIn, select=True, fraction=fraction,
                 low=psLow, center=psCenter, high=psHigh, ps=ps)
em.plot_fit(truths=parTruth, FigAx=(fig, axarr[1]),
            burnin=burnIn, select=True, fraction=fraction,
            low=psLow, center=psCenter, high=psHigh, ps=ps)
plt.savefig("{0}_result.png".format(targname), bbox_inches="tight")
plt.close()
print("Best fit plot finished!")