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
pc10   = 3.08567758e19 #cm
Lsun   = 3.828e33 #erg/s
mJy    = 1e26 # From erg/s/cm^2/Hz to mJy

#Generate the templates of stellar emission templates
logMs = 0.0
ageList = np.logspace(np.log10(0.05), 1.0, 100)
waveLim = [5e2, 1e7]
#ageList = np.array([0.5, 1.0, 3.0, 5.0, 9.0])
#model_name = "bc03_ssp_z_0.02_chab.model"
#model_name = "bc03_exp_1.0_z_0.02_chab.model"
modelList = [
    "bc03_ssp_z_0.02_chab.model",
    "bc03_burst_0.1_z_0.02_chab.model",
    "bc03_exp_0.1_z_0.02_chab.model",
    "bc03_exp_1.0_z_0.02_chab.model",
    "bc03_const_1.0_tV_0.2_z_0.02_chab.model",
    "bc03_const_1.0_tV_5.0_z_0.02_chab.model",
]
nAge = len(ageList)
nModel = len(modelList)
sedDict= {}
for model_name in modelList:
    sedDict[model_name] = []
    for age in ageList:
        sedDict[model_name].append(extractSED(age, 10**logMs, model_name=model_name))

#Interpolate with KD tree and spline interpolation
XList = []
tckList = []
for loop_m in range(nModel):
    model_name = modelList[loop_m]
    print "Constructing: {0}".format(model_name)
    for loop in range(nAge):
        age  = ageList[loop]
        (wave, flux) = sedDict[model_name][loop]
        fltr = (wave > waveLim[0]) & (wave < waveLim[1])
        wave = wave[fltr] * 1e-4 # Convert to micron
        flux = flux[fltr] * (4.0 * np.pi * pc10**2.) # Convert to erg/s/Hz
        tck = splrep(wave, flux)
        tckList.append(tck)
        XList.append([age, loop_m])
kdt = KDTree(XList)
print("Interpolation finishes!")
modelInfo = {
    "age": ageList,
    "model_name": modelList
}
parFormat = ["age", "SFH"]
readMe = '''
The stellar emission templates are generated with EzGal.
The units of the wavelength and flux are angstrom and mJy.
This template uses different SFHs coded from 0 to {0}.
'''.format(nModel-1)
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
fp = open("bc03_sps_cha_kdt.tmplt", "w")
pickle.dump(templateDict, fp)
fp.close()

#Test
cl = ['r', 'g', 'b', 'y', 'c']
for loop in range(len(cl)):
    pars = [ageList[loop], 0]
    wave, flux = sedDict["bc03_ssp_z_0.02_chab.model"][loop]
    fltr = (wave > waveLim[0]) & (wave < waveLim[1])
    wave = wave[fltr] * 1e-4
    flux = flux[fltr] * (4.0 * np.pi * pc10**2.)
    f = t(wave, pars)
    plt.plot(wave, f, color=cl[loop], linestyle="--")
    plt.plot(wave, flux, color=cl[loop], linestyle=":")
#plt.ylim([1e14, 1e20])
plt.xscale("log")
plt.yscale("log")
plt.show()
