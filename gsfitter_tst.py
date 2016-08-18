import h5py
import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from pprint import pprint
import cPickle as pickle
from scipy.interpolate import interp1d
import ndiminterpolation as ndip
import rel_SED_Toolkit as sedt
from fitter import basicclass as bc
from fitter import bandfunc   as bf
from fitter.sed import model_functions as sedmf
from fitter.sed import fit_functions   as sedff
from fitter.sed import sedclass        as sedsc

ls_mic = 2.99792458e14 #micron/s

logLFunc = sedff.logLFunc_SED
funcLib = sedmf.funcLib
inputModelDict = sedmf.inputModelDict
Model2Data = sedff.Model2Data

#Load SED data#
#-------------#
# Create the sedData
## Read in the data
targname = 'PG1612+261' #'PG0050+124'
sedFile  = '/Users/jinyi/Work/PG_QSO/sobt/SEDs/{0}_cbr.sed'.format(targname)
redshift = 0.061
sedRng   = [3, 13]
spcRng   = [13, None]
spcRebin = 1.
sedPck = sedt.Load_SED(sedFile, sedRng, spcRng, spcRebin)
sed = sedPck['sed_cb']
nsed = sedRng[1] - sedRng[0]
wave = sed[0]
flux = sed[1]
sigma = sed[2]
sedwave = sedPck['sed'][0]
sedflux = sedPck['sed'][1]
sedsigma = sedPck['sed'][2]
spcwave = sedPck['spc'][0]
spcflux = sedPck['spc'][1]
spcsigma = sedPck['spc'][2]

## Put into the sedData
bandList = ['w1', 'w2', 'w3', 'w4', 'PACS_70', 'PACS_100', 'PACS_160', 'SPIRE_250', 'SPIRE_350', 'SPIRE_500']
sedDataType = ['name', 'wavelength', 'flux', 'error', 'flag']
sedflag = np.ones_like(sedwave)
phtData = {'WISE&Herschel': bc.DiscreteSet(bandList, sedwave, sedflux, sedsigma, sedflag, sedDataType)}
spcflag = np.ones_like(spcwave)
spcDataType = ['wavelength', 'flux', 'error', 'flag']
spcData = {'IRS': bc.ContinueSet(spcwave, spcflux, spcsigma, spcflag, spcDataType)}
#spcData = {}
sedData = sedsc.SedClass(targname, redshift, phtDict=phtData, spcDict=spcData)
DL = sedData.dl
## Add the fake photometric data from the Spitzer spectrum
spitzerBandInfo = {
    's1': [6.0, 6.7],
    's2': [7.0, 9.0],
    's3': [10.0, 14.0],
    's4': [16.0, 20.0],
    's5': [24.0, 28.0]
}
spitzerBandDict = OrderedDict()
spitzerBandList = ['s1', 's2', 's3','s4', 's5']
for n in range(len(spitzerBandList)):
    bandName = spitzerBandList[n]
    waveMin = spitzerBandInfo[bandName][0]
    waveMax = spitzerBandInfo[bandName][1]
    fltr = (spcwave<waveMax) & (spcwave>waveMin)
    if np.sum(fltr) == 0:
        raise ValueError('The band range is incorrect!')
    bandWave = spcwave[fltr]
    bandRsr = np.ones_like(bandWave)
    spitzerBandDict[bandName] = sedsc.BandPass(bandWave, bandRsr, bandName=bandName)
sedData.set_bandpass(spitzerBandDict)
#print sedData.get_bandpass()
bandWaveList = []
bandFluxList = []
for bandName in spitzerBandList:
    bandWave, bandFlux = sedData.filtering(bandName, spcwave, spcflux)
    bandWaveList.append(bandWave)
    bandFluxList.append(bandFlux)
bandWaveList = np.array(bandWaveList)
bandFluxList = np.array(bandFluxList)
bandSigmaList = bandFluxList * 0.05
bandFlagList = np.ones_like(bandWaveList)
spitzerPhtData = {'Spitzer': bc.DiscreteSet(spitzerBandList, bandWaveList,
                                            bandFluxList, bandSigmaList,
                                            bandFlagList, sedDataType)}
sedData.add_DiscreteSet(spitzerPhtData)
## Plot the data
#fig, ax = sedData.plot_sed()
#ax.legend()
#plt.show()

# Load the bandpass
## Load WISE bandpass
wisePath = '/Users/jinyi/Work/PG_QSO/filters/wise/'
wiseBandDict = OrderedDict()
bandCenterList = [3.353, 4.603, 11.561, 22.088] #Isophotal wavelength
for n in range(4):
    bandName = 'w{0}'.format(n+1)
    bandFile = '{0}.dat'.format(bandName)
    bandPck = np.genfromtxt(wisePath+bandFile)
    bandWave = bandPck[:, 0]
    bandRsr = bandPck[:, 1]
    bandCenter = bandCenterList[n]
    wiseBandDict[bandName] = sedsc.BandPass(bandWave, bandRsr, bandCenter, bandName=bandName)
sedData.add_bandpass(wiseBandDict)
## Load Herschel bandpass
fp = open('/Users/jinyi/Work/PG_QSO/filters/herschel/herschel_bandpass.dict', 'r')
herschelBands = pickle.load(fp)
fp.close()
herschelBandList = herschelBands.keys()
herschelBandList.remove('README')
#print herschelBandList
herschelBandDict = OrderedDict()
for bandName in herschelBandList:
    #Be careful with the sequence!
    bandWave = ls_mic / herschelBands[bandName]['bandpass'][0][::-1]
    bandRsr = herschelBands[bandName]['bandpass'][1][::-1]
    bandCenter = herschelBands[bandName]['wave0']
    herschelBandDict[bandName] = sedsc.BandPass(bandWave, bandRsr, bandCenter,
                                             bandFunc=bf.BandFunc_intp,
                                             bandName=bandName)
sedData.add_bandpass(herschelBandDict)

#Build up the model#
#------------------#
## Load the templates

### CLUMPY template
clumpyFile = '/Users/jinyi/Work/PG_QSO/templates/clumpy_models_201410_tvavg.hdf5'
h = h5py.File(clumpyFile,'r')
theta = [np.unique(h[par][:]) for par in ('i','tv','q','N0','sig','Y','wave')]
data = h['flux_tor'].value
wave_tmpl = h['wave'].value
ip = ndip.NdimInterpolation(data,theta)

### DL07 template
fp = open('dl07_intp.dict', 'r')
tmpl_dl07_inpt = pickle.load(fp)
fp.close()
uminList = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.80, 1.00, 1.20,
            1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 7.00, 8.00, 10.0, 12.0,
            15.0, 20.0, 25.0]
umaxList = [1e6]
qpahList = [0.47, 1.12, 1.77, 2.50, 3.19, 3.90, 4.58, 0.75, 1.49, 2.37, 0.10]
waveModel = 10**np.linspace(0, 3, 500)

### Build the model
parAddDict_all = {
    'DL': DL,
    'tmpl_dl07': tmpl_dl07_inpt,
    'TORUS_tmpl_ip': ip
}
modelDict = {}
modelNameList = inputModelDict.keys()
for modelName in modelNameList:
    funcName = inputModelDict[modelName]['function']
    funcInfo = funcLib[funcName]
    xName = funcInfo['x_name']
    parFitList = funcInfo['param_fit']
    parAddList = funcInfo['param_add']
    parFitDict = {}
    parAddDict = {}
    for parName in parFitList:
        parFitDict[parName] = inputModelDict[modelName][parName]
    for parName in parAddList:
        parAddDict[parName] = parAddDict_all[parName]
    modelDict[modelName] = bc.ModelFunction(funcInfo['function'], xName, parFitDict, parAddDict)
sedModel = bc.ModelCombiner(modelDict)
#Plot the
#fig, ax = sedModel.plot(waveModel)
#plt.show()

#Fit with DNest4#
#---------------#

## Create the DNest4Model
dn4m = bc.DNest4Model(sedData, sedModel, logLFunc)
