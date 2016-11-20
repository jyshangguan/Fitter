import numpy as np
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from scipy.interpolate import splev

class Template(object):
    """
    This is the object of a model template.
    """

    def __init__(self, tckList, kdTree, parList, modelInfo={}, parFormat=[], readMe=""):
        self.__tckList   = tckList
        self.__kdTree    = kdTree
        self.__parList   = parList
        self.__modelInfo = modelInfo
        self.__parFormat = parFormat
        self._readMe    = readMe

    def __call__(self, x, pars):
        """
        Return the interpolation result of the template nearest the input
        parameters.
        """
        x = np.array(x)
        ind = np.squeeze(self.__kdTree.query(np.atleast_2d(pars), return_distance=False))
        tck = self.__tckList[ind]
        return splev(x, tck)

    def get_nearestParameters(self, pars):
        """
        Return the nearest template parameters to the input parameters.
        """
        ind = np.squeeze(self.__kdTree.query(np.atleast_2d(pars), return_distance=False))
        return self.__parList[ind]

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, dict):
        self.__dict__ = dict

    def get_parList(self):
        return self.__parList

    def get_modelInfo(self):
        return self.__modelInfo

    def get_parFormat(self):
        return self.__parFormat

    def readme(self):
        return self._readMe

def PCA_decompose(X, n_components, **kwargs):
    """
    Use PCA to decompose the input templates.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input templates to decompose.
    n_components : float
        The number of components kept.
    **kwargs : dict
        The arguments of PCA function.

    Returns
    -------
    results : dict
        X_t : array-like, shape (n_samples, n_components)
            The decomposed array of the input array.
        cmp : array-like, shape (n_components, n_features)
            The components decomposed by PCA.

    Notes
    -----
    None.
    """
    pca = PCA(n_components=n_components, **kwargs)
    X_t = pca.fit_transform(X)
    cmp = pca.components_
    evr = pca.explained_variance_ratio_
    results = {
        "X_t": X_t,
        "components": cmp,
        "evr": evr
    }
    return results

def PCA_recover(idx, encoder, decoder):
    """
    Recover the PCA decomposed array.

    Parameters
    ----------
    idx : float
        The index of the template to be recovered.
    encoder : HDF5 dataset
        The decomposed results of the original data.
    decoder : HDF5 dataset
        The principle components of the original data.

    Returns
    -------
    results : float array
        The recovered data array.

    Notes
    -----
    None.
    """
    nSamples    = encoder.attrs["nSamples"]
    nComponents = encoder.attrs["nComponents"]
    nFeatures   = decoder.attrs["nFeatures"]
    weight = encoder.value[idx, :]
    components = decoder.value
    result = np.zeros(nFeatures)
    for loop in range(nComponents):
        result += weight[loop] * components[loop, :]
    return result

if __name__ == "__main__":
    X = np.array([[-1, -1, -1], [-2, -1, 1], [-1, -2, 0], [1, 1, 2], [2, 1, 0], [1, 2, -1]])
    results = PCA_decompose(X, 3)
    X_t = results["X_t"]
    cmp = results["components"]
    print X_t
    print cmp
