from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import sys
import types
import importlib
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import rel_SED_Toolkit as sedt
from sedfit.fitter import basicclass as bc
from sedfit.fitter import bandfunc   as bf
from sedfit.fitter import mcmc
from sedfit import sedclass as sedsc
from sedfit import model_functions as sedmf
from collections import OrderedDict
from optparse import OptionParser
ls_mic = 2.99792458e14 #micron/s

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
im = importlib.import_module(moduleName.split(".")[0])


################################################################################
#                                    Data                                      #
################################################################################
targname = im.targname
sedFile  = im.sedPath + "{0}_rest.csed".format(targname)
sedRng = im.sedRng
sedPck = sedt.Load_SED(sedFile, sedRng, im.spcRng, im.spcRebin)
sed = sedPck["sed_cb"]
nsed = sedRng[1] - sedRng[0]
wave = sed[0]
flux = sed[1]
sigma = sed[2]
sedwave = sedPck["sed"][0]
sedflux = sedPck["sed"][1]
sedsigma = sedPck["sed"][2]
spcwave = sedPck["spc"][0]
spcflux = sedPck["spc"][1]
spcsigma = sedPck["spc"][2]
print("#--------------------------------#")
print("Target: {0}".format(targname))
print("SED file: {0}".format(sedFile))
print("#--------------------------------#")

## Put into the sedData
bandList = ["j", "h", "ks", "w1", "w2", "w3", "w4", "PACS_70", "PACS_100", "PACS_160", "SPIRE_250", "SPIRE_350", "SPIRE_500"]
sedDataType = ["name", "wavelength", "flux", "error", "flag"]
sedflag = np.ones_like(sedwave)
phtData = {"2M&W&H": bc.DiscreteSet(bandList, sedwave, sedflux, sedsigma, sedflag, sedDataType)}
spcflag = np.ones_like(spcwave)
spcDataType = ["wavelength", "flux", "error", "flag"]
spcData = {"IRS": bc.ContinueSet(spcwave, spcflux, spcsigma, spcflag, spcDataType)}
sedData = sedsc.SedClass(targname, im.redshift, phtDict=phtData, spcDict=spcData)

# Load the bandpass
## Load 2MASS bandpass
bandpass_dir = im.bandpass_dir
massPath = bandpass_dir+"2mass/"
massBandDict = OrderedDict()
bandNameList = ["j", "h", "ks"]
bandCenterList = [1.235, 1.662, 2.159] #Isophotal wavelength
for n in range(3):
    bandName = bandNameList[n]
    bandFile = "{0}.dat".format(bandName)
    bandPck = np.genfromtxt(massPath+bandFile, skip_header=1)
    bandWave = bandPck[:, 0]
    bandRsr = bandPck[:, 1]
    bandCenter = bandCenterList[n]
    massBandDict[bandName] = sedsc.BandPass(bandWave, bandRsr, bandCenter, bandName=bandName)
sedData.add_bandpass(massBandDict)
## Load WISE bandpass
wisePath = bandpass_dir+"wise/"
wiseBandDict = OrderedDict()
bandCenterList = [3.353, 4.603, 11.561, 22.088] #Isophotal wavelength
for n in range(4):
    bandName = "w{0}".format(n+1)
    bandFile = "{0}.dat".format(bandName)
    bandPck = np.genfromtxt(wisePath+bandFile)
    bandWave = bandPck[:, 0]
    bandRsr = bandPck[:, 1]
    bandCenter = bandCenterList[n]
    wiseBandDict[bandName] = sedsc.BandPass(bandWave, bandRsr, bandCenter, bandName=bandName)
sedData.add_bandpass(wiseBandDict)
## Load Herschel bandpass
fp = open(bandpass_dir+"herschel/herschel_bandpass.dict", "r")
herschelBands = pickle.load(fp)
fp.close()
herschelBandList = herschelBands.keys()
herschelBandList.remove("README")
#print herschelBandList
herschelBandDict = OrderedDict()
for bandName in herschelBandList:
    #Be careful with the sequence!
    bandWave = ls_mic / herschelBands[bandName]["bandpass"][0][::-1]
    bandRsr = herschelBands[bandName]["bandpass"][1][::-1]
    bandCenter = herschelBands[bandName]["wave0"]
    herschelBandDict[bandName] = sedsc.BandPass(bandWave, bandRsr, bandCenter,
                                             bandFunc=bf.BandFunc_Herschel,
                                             bandName=bandName)
sedData.add_bandpass(herschelBandDict)


################################################################################
#                                   Model                                      #
################################################################################
modelDict = im.modelDict
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
parAddDict_all = {
    "DL": sedData.dl,
}
funcLib    = sedmf.funcLib
waveModel = 10**np.linspace(0.0, 3.0, 1000)
sedModel = bc.Model_Generator(modelDict, funcLib, waveModel, parAddDict_all)
parAllList = sedModel.get_parVaryList()
parAllList.append(np.log(-np.inf))
parAllList.append(-np.inf)
parAllList.append(-5)
parTruth = im.parTruth   #Whether to provide the truth of the model
modelUnct = im.modelUnct #Whether to consider the model uncertainty in the fitting
ndim = len(parAllList)


################################################################################
#                                   emcee                                      #
################################################################################
#Fit with MCMC#
#-------------#
emceeDict = im.emceeDict
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
ppDict   = im.ppDict
psLow    = ppDict["low"]
psCenter = ppDict["center"]
psHigh   = ppDict["high"]
nuisance = ppDict["nuisance"]

print("#--------------------------------#")
print("emcee Info:")
for keys in emceeDict.keys():
    print("{0}: {1}".format(keys, emceeDict[keys]))
print("#--------------------------------#")
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
