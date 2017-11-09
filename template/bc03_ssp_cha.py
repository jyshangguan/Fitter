import numpy as np
from sedfit.fitter.template import Template
from sklearn.neighbors import KDTree
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
import cPickle as pickle
from sgPhot import extractSED
ls_mic = 2.99792458e14 #micron/s
ls_aa  = 2.99792458e18 #aa/s
Mpc    = 3.08567758e24 #cm
Lsun   = 3.828e33 #erg/s
mJy    = 1e26 # From erg/s/cm^2/Hz to mJy

#Generate the templates of stellar emission templates
logMs = 0.0
ageList = np.logspace(np.log10(0.05), 1.0, 100)
waveLim = [5e2, 1e7]
#ageList = np.array([0.5, 1.0, 3.0, 5.0, 9.0])
model_name="bc03_ssp_z_0.02_chab.model"
nAge = len(ageList)
sedList = []
for age in ageList:
    sedList.append(extractSED(age, 10**logMs, model_name=model_name))

#Interpolate with KD tree and spline interpolation
XList = []
tckList = []
for loop in range(nAge):
    age  = ageList[loop]
    (wave, flux) = sedList[loop]
    fltr = (wave > waveLim[0]) & (wave < waveLim[1])
    wave = wave[fltr]
    flux = flux[fltr] * mJy
    tck = splrep(wave, flux)
    tckList.append(tck)
    XList.append([age])
kdt = KDTree(XList)
print("Interpolation finishes!")
modelInfo = {
    "age": ageList,
}
parFormat = ["age"]
readMe = '''
The stellar emission templates are generated with EzGal.
The units of the wavelength and flux are angstrom and mJy.
'''
templateDict = {
    "tckList": tckList,
    "kdTree": kdt,
    "parList": XList,
    "modelInfo": modelInfo,
    "parFormat": parFormat,
    "readMe": readMe
}
t = Template(tckList=tckList, kdTree=kdt, parList=XList, modelInfo=modelInfo,
             parFormat=parFormat, readMe=readMe)
t = Template(**templateDict)
fp = open("bc03_ssp_cha_kdt.tmplt", "w")
pickle.dump(templateDict, fp)
fp.close()

#Test
cl = ['r', 'g', 'b', 'y', 'c']
for loop in range(len(cl)):
    pars = [ageList[loop]]
    wave, flux = sedList[loop]
    fltr = (wave > waveLim[0]) & (wave < waveLim[1])
    wave = wave[fltr]
    flux = flux[fltr] * mJy
    f = t(wave, pars)
    plt.plot(wave, f, color=cl[loop], linestyle="--")
    plt.plot(wave, flux, color=cl[loop], linestyle=":")
#plt.ylim([1e14, 1e20])
plt.xscale("log")
plt.yscale("log")
plt.show()
