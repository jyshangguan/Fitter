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

funcLib = sedmf.funcLib
inputModelDict = sedmf.inputModelDict
Model2Data = sedff.Model2Data
Parameters_Init = sedff.Parameters_Init
Parameters_Dump = sedff.Parameters_Dump
Parameters_Load = sedff.Parameters_Load
ChiSquare_LMFIT = sedff.ChiSquare_LMFIT

#Load SED data
##Read in the data
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

##Put into the sedData
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
##Add the fake photometric data from the Spitzer spectrum
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
##Plot the data
fig, ax = sedData.plot_sed()
ax.legend()
plt.show()

#Load the bandpass
##Load WISE bandpass
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
##Load Herschel bandpass
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

#
