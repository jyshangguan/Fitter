import h5py
import numpy as np
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
    if counter > 100:
        break

fnuList = np.array(fnuList)
nComponents = 20
pcaResults = PCA_decompose(fnuList, nComponents)
print fnuList.shape
X_t = pcaResults["X_t"]
cmp = pcaResults["components"]
print X_t.shape
print cmp.shape

f = h5py.File("template/clumpy_pca.hdf5", "w")
encoder = f.create_dataset("encoder", (nSamples, nComponents), dtype="float")
decoder = f.create_dataset("decoder", (nComponents, nFeatures), dtype="float")
#encoder[...] = X_t
encoder[0:102, :] = X_t
decoder[...] = cmp
f.attrs["README"] = """
    This file saves the PCA decomposed CLUMPY model. The recovered data is
    log10(fnu) of the model. The integrated flux is normalised to 1.
    """
encoder.attrs["README"] = "The decomposed templates."
decoder.attrs["README"] = "The PCA components of the model."
print fnuList
print encoder.shape
print encoder.value
print decoder.shape
print decoder.value
f.flush()
f.close()
