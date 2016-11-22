import os
import numpy as np
import cPickle as pickle
from astropy.table import Table
from collections import OrderedDict
from sedfit.fitter import basicclass as bc
from sedfit import model_functions as sedmf

##Find all the result files
resultPath = "/Volumes/Transcend/Work/PG_MCMC/pg_sil/"
bestfitList = []
fitrsList   = []
for f in os.listdir(resultPath):
    if f.endswith(".txt"):
        bestfitList.append(f)
    if f.endswith(".fitrs"):
        fitrsList.append(f)
nTargets = len(fitrsList)

def AddDict(targetDict, quantName, quant):
    """
    To add a quantity into the target dict.
    """
    if quantName in targetDict.keys():
        targetDict[quantName].append(quant)
    else:
        targetDict[quantName] = [quant]
    return None

#The function should be specified for different models
def CorrectParameters(pars):
    """
    Correct the parameters for the functions with discrete parameters.
    """
    nCut = 10
    parUnuse = list(pars[0:nCut])
    parInuse = pars[nCut:]
    parKws = {
        "logumin": parInuse[0],
        "logumax": 6,
        "qpah": parInuse[1],
        "gamma": parInuse[2],
        "logMd": parInuse[3]
    }
    pD = sedmf.DL07_PosPar(**parKws)
    parCorrect = [pD["logumin"], pD["qpah"], pD["gamma"], pD["logMd"]]
    #print parCorrect
    parCombine = np.array(parUnuse + parCorrect)
    return parCombine

def Luminosity_Integrate(flux, wave, DL, waveRange=[8.0, 1e3]):
    """
    Calculate the integrated luminosity of input SED.
    """
    ls_mic = 2.99792458e14 #unit: micron/s
    Mpc = 3.08567758e24 #unit: cm
    mJy = 1e26 #unit: erg/s/cm^2/Hz
    nu = ls_mic / wave
    fltr = (wave > waveRange[0]) & (wave < waveRange[1])
    F = -1.0 * np.trapz(flux[fltr], nu[fltr]) / mJy #unit: erg/s/cm^2
    L = F * 4.0*np.pi * (DL * Mpc)**2.0
    return L

##Build the bestfit model
resultDict = OrderedDict()

#loop_f = 0
#nTargets = 5
for loop_f in range(nTargets):
    f = open(resultPath+fitrsList[loop_f], "r")
    fitrs = pickle.load(f)
    f.close()
    #bestfit = np.genfromtxt(resultPath+bestfitList[loop_f], skip_header=1, dtype=None)
    #print bestfit

    dataPck = fitrs["dataPck"]
    targname = dataPck["targname"]
    print targname
    AddDict(resultDict, "Name", targname)

    modelPck = fitrs["modelPck"]
    funcLib = sedmf.funcLib
    modelDict = modelPck["modelDict"]
    waveModel = modelPck["waveModel"]
    parAddDict_all = modelPck["parAddDict_all"]
    sedModel = bc.Model_Generator(modelDict, funcLib, waveModel, parAddDict_all)
    parName = sedModel.get_parVaryNames(latex=False)
    nPar = len(parName)
    #print parName

    ppDict = fitrs["ppDict"]
    ps = fitrs["posterior_sample"]
    psLow    = ppDict["low"]
    psCenter = ppDict["center"]
    psHigh   = ppDict["high"]
    parRange = np.percentile(ps, [psLow, psCenter, psHigh], axis=0)
    parL = CorrectParameters(parRange[0, 0:nPar])
    parC = CorrectParameters(parRange[1, 0:nPar])
    parU = CorrectParameters(parRange[2, 0:nPar])

    for loop_p in range(nPar):
        pn = parName[loop_p]
        AddDict(resultDict, pn+"_l", parL[loop_p])
        AddDict(resultDict, pn+"_c", parC[loop_p])
        AddDict(resultDict, pn+"_u", parU[loop_p])

    ##Calculate the integrated luminosities
    parListDict = OrderedDict(
        [("L", parL),
        ("C", parC),
        ("U", parU)]
    )
    DL = parAddDict_all["DL"]
    modelName = modelDict.keys()
    #Component integrated luminosity
    for loop_m in range(len(modelName)):
        mn = modelName[loop_m]
        for scp in parListDict.keys():
            parList = parListDict[scp]
            sedModel.updateParList(parList)
            fluxComps = sedModel.componentResult()
            flux = fluxComps[mn]
            lum  = Luminosity_Integrate(flux, waveModel, DL, waveRange=[8.0, 1e3])
            AddDict(resultDict, "L{0}_{1}".format(mn.split(" ")[0], scp), lum)
    #Total integrated luminosity
    for scp in parListDict.keys():
        parList = parListDict[scp]
        sedModel.updateParList(parList)
        fluxTotal = sedModel.combineResult()
        Ltotal = Luminosity_Integrate(fluxTotal, waveModel, DL, waveRange=[8.0, 1e3])
        AddDict(resultDict, "Ltotal_{0}".format(scp), Ltotal)

tNameList  = resultDict.keys()
tQuantList = []
for tn in tNameList:
    tQuantList.append(resultDict[tn])
resultTable = Table(tQuantList, names=tNameList)
resultTable.write(resultPath+"pg_sil.ipac", format="ascii.ipac")
