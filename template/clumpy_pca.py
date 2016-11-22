#This script use the PCA method to decompose the CLUMPY templates. It is found
#that the normalised templates are not very well recovered. Therefore, we
#decompose the original ones from "clumpy_models_201410_tvavg.hdf5".
#The aim of this decomposition is to reduce the data file size while keeping the
#accuracy of the templates.
#We first take the log10 of the templates and then subtract the mean of the
#logarithm fluxes from each of the template. The PCA decomposition is applied to
#the results. The steps come from Han&Han ApJ, 749, 123, 2012.

import h5py
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from sklearn.neighbors import KDTree
from sedfit.fitter.template import PCA_decompose

##Load the data for PCA and KDTree
#h = h5py.File('/Users/jinyi/Work/PG_QSO/templates/clumpy_fnu_norm.hdf5')
h = h5py.File('/Users/jinyi/Work/PG_QSO/templates/clumpy_models_201410_tvavg.hdf5')
h_q = h['q'].value
h_N0 = h['N0'].value
h_tv = h['tv'].value
h_sig = h['sig'].value
h_Y = h['Y'].value
h_i = h['i'].value
wave = h['wave'].value
flux_tor = h["flux_tor"].value

nSamples  = len(flux_tor)
nFeatures = len(flux_tor[0])
print nSamples, nFeatures
parList = []
fnuList = []
for counter in range(nSamples):
    q   = h_q[counter]
    N0  = h_N0[counter]
    tv  = h_tv[counter]
    sig = h_sig[counter]
    Y   = h_Y[counter]
    i   = h_i[counter]
    logfnu = np.log10(flux_tor[counter]) #For better behaviour of PCA decomposition.
    parList.append([q, N0, tv, sig, Y, i])
    fnuList.append(logfnu)

fnuList = np.array(fnuList)
fnuMean = np.mean(fnuList, axis=0)
ipList = np.zeros_like(fnuList)
for nf in range(nFeatures):
    ipList[:, nf] = fnuList[:, nf] - fnuMean[nf] #For better behaviour of PCA decomposition.

##PCA decomposition
nComponents = 16
pcaResults = PCA_decompose(ipList, nComponents, svd_solver="full")
X_t = pcaResults["X_t"]
cmp = pcaResults["components"]
evr = pcaResults["evr"]
sEvr = sum(evr)
print "PCA finish! {0} components explain {1} of the variance.".format(nComponents, sEvr)

##Save the PCA decomposed results
#f = h5py.File("template/clumpy_pca.hdf5", "w")
f = h5py.File("template/clumpy_pca.hdf5", "w")
wave_f = f.create_dataset("wave", (nFeatures,), dtype="float")
fnu_mean = f.create_dataset("fnu_mean", (nFeatures,), dtype="float")
encoder = f.create_dataset("encoder", (nSamples, nComponents), dtype="float")
decoder = f.create_dataset("decoder", (nComponents, nFeatures), dtype="float")
wave_f[...] = wave
fnu_mean[...] = fnuMean
encoder[...] = X_t
decoder[...] = cmp
'''
f.attrs["README"] = """
    This file saves the PCA decomposed CLUMPY model. The recovered data is
    the mean subtracted log10(fnu) of the model. The integrated flux is
    normalised to 1.
    To recover the fnu, one should first recover the input array with the
    PCA encoder and decoder. Add the mean SED and then use it as the
    exponent of 10.
    There are {0} principle components used.
    """.format(nComponents)
'''
f.attrs["README"] = """
    This file saves the PCA decomposed CLUMPY model. The recovered data is
    the mean subtracted log10(fnu) of the model. The templates are not normalised.
    To recover the fnu, one should first recover the input array with the
    PCA encoder and decoder. Add the mean SED and then use it as the
    exponent of 10.
    There are {0} principle components used.
    """.format(nComponents)
wave_f.attrs["README"] = "The corresponding wavelength array."
fnu_mean.attrs["README"] = "The mean SED of log10(fnu)."
encoder.attrs["README"] = "The decomposed templates."
encoder.attrs["nSamples"] = nSamples
encoder.attrs["nComponents"] = nComponents
decoder.attrs["README"] = """
    The PCA components of the model. The total {0} components explain {1} of
    the variance .
    """.format(nComponents, sEvr)
decoder.attrs["nComponents"] = nComponents
decoder.attrs["nFeatures"] = nFeatures

f.flush()
f.close()

##Build up the KDTree
kdt = KDTree(parList)
print "The KDTree is built up!"
modelInfo = {
    "q": np.unique(h_q),
    "N0": np.unique(h_N0),
    "tv": np.unique(h_tv),
    "sigma": np.unique(h_sig),
    "Y": np.unique(h_Y),
    "i": np.unique(h_i),
    "wavelength": wave,
}

parFormat = ["q", "N0", "tv", "sigma", "Y", "i"]
readMe = '''
    This template is from: http://www.pa.uky.edu/clumpy/
    The interpolation is tested well!
    '''
templateDict = {
    "pcaFile": "clumpy_unnorm_pca.hdf5",
    "kdTree": kdt,
    "parList": parList,
    "modelInfo": modelInfo,
    "parFormat": parFormat,
    "readMe": readMe
}

fp = open("clumpy_kdt.tmplt", "w")
pickle.dump(templateDict, fp)
fp.close()
