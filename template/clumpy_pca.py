import h5py
import numpy as np
import matplotlib.pyplot as plt
from sedfit.fitter.template import PCA_decompose

h = h5py.File('/Users/jinyi/Work/PG_QSO/templates/clumpy_fnu_norm.hdf5')
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
    logfnu = np.log10(flux_tor[counter])
    parList.append([q, N0, tv, sig, Y, i])
    fnuList.append(logfnu)
    #if counter > 100:
    #    break

fnuList = np.array(fnuList)
fnuMean = np.mean(fnuList, axis=0)
ipList = np.zeros_like(fnuList)
for nf in range(nFeatures):
    ipList[:, nf] = fnuList[:, nf] - fnuMean[nf]
#plt.plot(wave, fnuMean)
#plt.xscale("log")
#plt.show()

nComponents = 16
pcaResults = PCA_decompose(ipList, nComponents, svd_solver="full")
X_t = pcaResults["X_t"]
cmp = pcaResults["components"]

f = h5py.File("template/clumpy_pca.hdf5", "w")
wave_f = f.create_dataset("wave", (nFeatures,), dtype="float")
fnu_mean = f.create_dataset("fnu_mean", (nFeatures,), dtype="float")
encoder = f.create_dataset("encoder", (nSamples, nComponents), dtype="float")
decoder = f.create_dataset("decoder", (nComponents, nFeatures), dtype="float")
wave_f[...] = wave
fnu_mean[...] = fnuMean
encoder[...] = X_t
#encoder[0:102, :] = X_t
decoder[...] = cmp
f.attrs["README"] = """
    This file saves the PCA decomposed CLUMPY model. The recovered data is
    the mean subtracted log10(fnu) of the model. The integrated flux is
    normalised to 1.
    To recover the fnu, one should first recover the input array with the
    PCA encoder and decoder. Add the mean SED and then use it as the
    exponent of 10.
    """
wave_f.attrs["README"] = "The corresponding wavelength array."
fnu_mean.attrs["README"] = "The mean SED of log10(fnu)."
encoder.attrs["README"] = "The decomposed templates."
encoder.attrs["nSamples"] = nSamples
encoder.attrs["nComponents"] = nComponents
decoder.attrs["README"] = "The PCA components of the model."
decoder.attrs["nComponents"] = nComponents
decoder.attrs["nFeatures"] = nFeatures

f.flush()
f.close()
