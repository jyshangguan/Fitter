import os
import h5py
import copy
import dnest4
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
targname = 'mock_dl07' #"mock_line"#'PG1612+261' #'PG0050+124'
sedFile  = "mock_dl07.sed" #"mock_line.sed"#'mock_mbb.sed' #'/Users/jinyi/Work/PG_QSO/sobt/SEDs/{0}_cbr.sed'.format(targname)
redshift = 0.2 #0.061
sedRng   = [0, 10]
spcRng   = [10, None]
#sedRng   = [3, 13]
#spcRng   = [13, None]
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
                                             bandFunc=bf.BandFunc_Herschel,
                                             bandName=bandName)
sedData.add_bandpass(herschelBandDict)

#Build up the model#
#------------------#

parAddDict_all = {
    'DL': DL,
}
waveModel = sedmf.waveModel
sedModel = bc.Model_Generator(inputModelDict, funcLib, waveModel, parAddDict_all)
parAllList = sedModel.get_parList()
#print inputModelDict
parNumber  = len(parAllList)
print("The model we are using:")
for modelName in inputModelDict.keys():
    print("{0}".format(modelName))
print("#---------------------#")


#Fit with DNest4#
#---------------#

## Create the DNest4Model
fp = open("{0}.dict".format(targname))
mockDict = pickle.load(fp)
fp.close()
fTrue = mockDict.get('f_add', None)
if fTrue is None:
    modelUnct = False
else:
    modelUnct = True #Whether to consider the model uncertainty in the fitting
    parNumber += 1
dn4m = bc.DNest4Model(sedData, sedModel, logLFunc, modelUnct)

"""
## Create a model object and a sampler
sampler = dnest4.DNest4Sampler(dn4m,
                               backend=dnest4.backends.CSVBackend(".",
                                                                  sep=" "))

## Set up the sampler. The first argument is max_num_levels
gen = sampler.sample(max_num_levels=30, num_steps=1000, new_level_interval=10000,
                      num_per_step=10000, thread_steps=100,
                      num_particles=5, lam=10, beta=100, seed=1234)

## Do the sampling (one iteration here = one particle save)
for i, sample in enumerate(gen):
    print("# Saved {k} particles.".format(k=(i+1)))


## Run the postprocessing
dnest4.postprocess()
#"""

#"""
#Process the posterior distributions of the parameters#
#-----------------------------------------------------#
##Fitting results
ps = np.loadtxt("posterior_sample.txt")
try:
    assert ps.shape[1] == parNumber
except:
    raise AssertionError("The posterior sample is problematic!")
##Calculate the optimized paramter values
parRangeList = map( lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                   zip(*np.percentile(ps, [16, 50, 84], axis=0, interpolation='nearest')) )
parRangeList = np.array(parRangeList)
parBfList = parRangeList[:, 0] #The best fit parameters
uncUlList = parRangeList[:, 1] #The upperlimits of the parameter uncertainties
uncLlList = parRangeList[:, 2] #The lowerlimits of the parameter uncertainties
parUlList = parBfList + uncUlList #The upperlimits of the parameters
parLlList = parBfList - uncLlList #The lowerlimits of the parameters
sedModel.updateParList(parBfList)
yFitBf = sedModel.combineResult()
yFitBF_cmp = sedModel.componentResult()
sedModel.updateParList(parUlList)
yFitUl = sedModel.combineResult()
yFitUl_cmp = sedModel.componentResult()
sedModel.updateParList(parLlList)
yFitLl = sedModel.combineResult()
yFitLl_cmp = sedModel.componentResult()

### Plot the Data
cmpList = mockDict['components']
parList = mockDict['parameters']
xModel  = mockDict['x_model']
loglTrue = mockDict['logl_true']

if modelUnct:
    parList.append( np.log(fTrue) )
for loop in range(len(parBfList)):
    p = parBfList[loop]
    u = uncUlList[loop]
    l = uncLlList[loop]
    t = parList[loop]
    print("{0}+{1}-{2}, {3}".format(p, u, l, t))

plt.close()
plt.figure()
cList = ['r', 'g', 'b', 'm', 'y', 'c']
plt.plot(waveModel, yFitBf, color="grey", linewidth=3.0)
plt.fill_between(waveModel, yFitLl, yFitUl, color="grey", alpha=0.1)
counter = 0
for modelName in sedModel._modelList:
    plt.plot(waveModel, yFitBF_cmp[modelName], color=cList[counter])
    plt.fill_between(waveModel, yFitLl_cmp[modelName], yFitUl_cmp[modelName],
                     color=cList[counter], alpha=0.1)
    counter += 1
plt.errorbar(wave, flux, yerr=sigma, fmt='.k', markersize=10, elinewidth=2, capsize=2)
yModel = np.zeros_like(xModel)
counter = 0
for y in cmpList:
    plt.plot(xModel, y, color=cList[counter], linestyle='--')
    yModel += y
    counter += 1
plt.plot(xModel, yModel, color='k', linewidth=1.5)
ymax = np.max(yModel)
plt.ylim([1e-2, ymax*2.0])
plt.xscale('log')
plt.yscale('log')
plt.savefig("{0}_result.png".format(targname))
plt.close()

## Rename the posterior sample file name#
#os.rename("")
#os.rename("posterior_sample.txt", "{0}_posterior.txt".format(targname))
#"""
