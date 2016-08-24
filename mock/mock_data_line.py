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
ls_mic = 2.99792458e14 #micron/s
inputModelDict = sedmf.inputModelDict
funcLib = sedmf.funcLib
Model2Data = sedff.Model2Data

#Generate the mock data#
#-----------------#
## Creat an SedClass object
targname = 'mock_line'
redshift = 0.2
bandList = ['w1', 'w2', 'w3', 'w4', 'PACS_70', 'PACS_100', 'PACS_160', 'SPIRE_250', 'SPIRE_350', 'SPIRE_500']
sedwave = np.array([3.353, 4.603, 11.561, 22.088, 70.0, 100.0, 160.0, 250.0, 350.0, 500.0])/(1+redshift)
fakedata = np.ones_like(sedwave)
spcwave = np.linspace(5.5, 38.0, 250)/(1+redshift)
spcfake = np.ones_like(spcwave)
phtData = {'WISE&Herschel': bc.DiscreteSet(bandList, sedwave, fakedata, fakedata, fakedata)}
spcData = {"Spitzer": bc.ContinueSet(spcwave, spcfake, spcfake, spcfake)}
sedData = sedsc.SedClass(targname, redshift, phtDict=phtData, spcDict=spcData)
wave = np.array(sedData.get_List("x"))
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

### Build the model
fAdd   = 0.05 #None
Ndata  = 500
xMax   = 600.0
waveModel = 10**np.linspace(0.0, np.log10(xMax), Ndata)
sedModel = bc.Model_Generator(inputModelDict, funcLib, waveModel)
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
    yObsr += np.abs(fAdd * yObsr) * np.random.randn(nBands)
yObsr += yErr * np.random.randn(nBands)

if not fAdd is None:
    s2 = yErr**2. + (yTrueBand * fAdd)**2.
else:
    s2 = yErr**2.
logL_dir = -0.5 * np.sum( (yObsr - yTrueBand)**2. / s2 + np.log(2 * np.pi * s2) )
print("logL_dir: {0:.3f}".format(logL_dir))
chisq = sedff.ChiSq(yObsr, yTrueBand, s2**0.5)
logL_sed = -0.5 * ( chisq + np.sum( np.log(2 * np.pi * s2) ) )
print("logL_sed: {0:.3f}".format(logL_sed))

model = {}
model['x_model']    = waveModel
model['x_obsr']     = sedwave
model['y_obsr']     = yObsr
model['y_err']      = yErr
model['y_true']     = yTrueBand
model['f_add']      = fAdd
model['parameters'] = parList
model['components'] = cmpList
model['logl_true']  = logL_dir
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
ymin = np.min(yTrueBand)
ymax = np.max(yTrueBand)
plt.xlim([1.0, 7e2])
plt.ylim([ymin/2.0, ymax*2.0])
plt.xscale("log")
plt.yscale("log")
plt.savefig("{0}.pdf".format(targname))
plt.show()
