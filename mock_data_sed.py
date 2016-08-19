import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from collections import OrderedDict
from fitter import bandfunc as bf
from fitter import basicclass as bc
from fitter.sed import sedclass as sedsc
from fitter.sed.model_functions import Modified_BlackBody
ls_mic = 2.99792458e14 #micron/s

#Generate the mock data#
#-----------------#
## Creat an SedClass object
targname = 'mock'
redshift = 0.2
bandList = ['w1', 'w2', 'w3', 'w4', 'PACS_70', 'PACS_100', 'PACS_160', 'SPIRE_250', 'SPIRE_350', 'SPIRE_500']
nBands = len(bandList)
sedwave = np.array([3.353, 4.603, 11.561, 22.088, 70.0, 100.0, 160.0, 250.0, 350.0, 500.0])
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

## Generate the mock data

fAdd   = None #0.1
logMList = [1.2, 4.0, 9.4]
TList = [641.8, 147.4, 26.1]
betaList = [1.7, 2.3, 2.0]
print( "#------Start fitting------#" )
parList = []
for loop in range(3):
    pTuple = (logMList[loop], TList[loop], betaList[loop])
    parList.append(pTuple)
    print("MBB{0}:logM ({1[0]}),  T ({1[1]}), beta ({1[2]})".format(loop, pTuple))
print( "# f_add: {0}".format(fAdd) )
print( "#-------------------------#" )

Ndata  = 500
xMax   = 500.0
x = 10**np.linspace(0.0, np.log10(xMax), Ndata)
yTrue = np.zeros(Ndata)
cmpList = []
for loop in range(3):
    logM, T, beta = parList[loop]
    yCmp = Modified_BlackBody(logM, T, beta, x, DL)
    yTrue += np.array(yCmp)
    cmpList.append(yCmp)
yTrueBand = sedData.model_pht(x, yTrue)
yTrueBand = np.array(yTrueBand)
#yErr = 5.0 + 30.0 * np.random.rand(nBands)
yErr = np.concatenate([1.0 * np.ones(4), 10.0*np.ones(3), 20.0*np.ones(3)])
yObsr = yTrueBand.copy()
if not fAdd is None:
    print("The model uncertainty is considered!")
    yObsr += np.abs(fAdd * yObsr) * np.random.rand(nBands)
yObsr += yErr * np.random.randn(nBands)
model = {}
model['x']          = x
model['x_obsr']     = sedwave
model['y_obsr']     = yObsr
model['y_err']      = yErr
model['parameters'] = parList
model['compnents']  = cmpList

fileName = "mbb_mock.dict"
fp = open(fileName, "w")
pickle.dump(model, fp)
fp.close()
print("{0} is saved!".format(fileName))
sedArray = np.array([sedwave, yObsr, yErr, np.ones(nBands)])
sedArray = np.transpose(sedArray)
sedFileName = "mock.sed"
fp = open(sedFileName, "w")
fp.write("wavelength\tflux\tsigma\n")
np.savetxt(fp, sedArray, fmt='%.3f', delimiter='\t')
fp.close()

fig = plt.figure()
plt.errorbar(sedwave, yObsr, yerr=yErr, fmt=".r")
plt.scatter(sedwave, yTrueBand, linewidth=1.5, color="k")
for y in cmpList:
    plt.plot(x, y, linestyle='--')
plt.plot(x, yTrue, color="k")
ymax = np.max(yTrue)
plt.xlim([1.0, 6e2])
plt.ylim([1e1, ymax*2.0])
plt.xscale("log")
plt.yscale("log")
plt.savefig("mbb_mock.pdf")
plt.close()
