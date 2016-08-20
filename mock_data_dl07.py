import h5py
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from collections import OrderedDict
import ndiminterpolation as ndip
from fitter import bandfunc as bf
from fitter import basicclass as bc
from fitter.sed import sedclass as sedsc
from fitter.sed import model_functions as sedmf
from fitter.sed import fit_functions as sedff
from fitter.sed.model_functions import Modified_BlackBody
ls_mic = 2.99792458e14 #micron/s
inputModelDict = sedmf.inputModelDict
funcLib = sedmf.funcLib

#Generate the mock data#
#-----------------#
## Creat an SedClass object
targname = 'mock_dl07'
redshift = 0.2
bandList = ['w1', 'w2', 'w3', 'w4', 'PACS_70', 'PACS_100', 'PACS_160', 'SPIRE_250', 'SPIRE_350', 'SPIRE_500']
nBands = len(bandList)
sedwave = np.array([3.353, 4.603, 11.561, 22.088, 70.0, 100.0, 160.0, 250.0, 350.0, 500.0])/(1+redshift)
fakedata = np.ones(nBands)
phtData = {'WISE&Herschel': bc.DiscreteSet(bandList, sedwave, fakedata, fakedata, fakedata)}
sedData = sedsc.SedClass(targname, redshift, phtDict=phtData)
DL = sedData.dl

## Load the bandpass
### Load WISE bandpass
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
### Load Herschel bandpass
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
                                             bandFunc=bf.BandFunc_Herschel,
                                             bandName=bandName)
sedData.add_bandpass(herschelBandDict)

## Build the model
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

### Build the model
parAddDict_all = {
    'DL': DL,
    'tmpl_dl07': tmpl_dl07_inpt,
    'TORUS_tmpl_ip': ip
}
modelDict = OrderedDict()
modelNameList = inputModelDict.keys()
for modelName in modelNameList:
    funcName = inputModelDict[modelName]['function']
    funcInfo = funcLib[funcName]
    xName = funcInfo['x_name']
    parFitList = funcInfo['param_fit']
    parAddList = funcInfo['param_add']
    parFitDict = OrderedDict()
    parAddDict = {}
    for parName in parFitList:
        parFitDict[parName] = inputModelDict[modelName][parName]
    for parName in parAddList:
        parAddDict[parName] = parAddDict_all[parName]
    modelDict[modelName] = bc.ModelFunction(funcInfo['function'], xName, parFitDict, parAddDict)

fAdd   = None #0.1
Ndata  = 500
xMax   = 600.0
waveModel = 10**np.linspace(0.0, np.log10(xMax), Ndata)
sedModel = bc.ModelCombiner(modelDict, waveModel)
parList = sedModel.get_parList()
#print inputModelDict
parNumber  = len(parList)

yTrue = sedModel.combineResult()
yDict = sedModel.componentResult()
cmpList = []
for modelName in yDict.keys():
    cmpList.append(yDict[modelName])

yTrueBand = sedData.model_pht(waveModel, yTrue)
yTrueBand = np.array(yTrueBand)
yErr = np.concatenate([1.0 * np.ones(4), 7.0*np.ones(3), 10.0*np.ones(3)])
yObsr = yTrueBand.copy()
if not fAdd is None:
    print("The model uncertainty is considered!")
    yObsr += np.abs(fAdd * yObsr) * np.random.rand(nBands)
yObsr += yErr * np.random.randn(nBands)
model = {}
model['x_model']    = waveModel
model['x_obsr']     = sedwave
model['y_obsr']     = yObsr
model['y_err']      = yErr
model['y_true']     = yTrueBand
model['parameters'] = parList
model['components']  = cmpList

logL_dir = -0.5 * np.sum( ((yObsr - yTrueBand)/yErr)**2. + np.log(2 * np.pi * yErr**2.) )
print("logL_dir: {0:.3f}".format(logL_dir))
chisq = sedff.ChiSq(yObsr, yTrueBand, yErr)
print chisq
logL_sed = -0.5 * ( chisq + np.sum( np.log(2 * np.pi * yErr**2.) ) )
print("logL_sed: {0:.3f}".format(logL_sed))

fileName = "{0}.dict".format(targname)
fp = open(fileName, "w")
pickle.dump(model, fp)
fp.close()
print("{0} is saved!".format(fileName))
sedArray = np.array([sedwave, yObsr, yErr, np.ones(nBands)])
sedArray = np.transpose(sedArray)
sedFileName = "{0}.sed".format(targname)
fp = open(sedFileName, "w")
fp.write("wavelength\tflux\tsigma\n")
np.savetxt(fp, sedArray, fmt='%.3f', delimiter='\t')
fp.close()

fig = plt.figure()
plt.errorbar(sedwave, yObsr, yerr=yErr, fmt=".r")
plt.scatter(sedwave, yTrueBand, linewidth=1.5, color="k")
for y in cmpList:
    plt.plot(waveModel, y, linestyle='--')
plt.plot(waveModel, yTrue, color="k")
ymax = np.max(yTrue)
plt.xlim([1.0, 7e2])
plt.ylim([1e-3, ymax*2.0])
plt.xscale("log")
plt.yscale("log")
plt.savefig("{0}.pdf".format(targname))
plt.close()