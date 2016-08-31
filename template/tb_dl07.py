import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
#from sedfit.fitter.template import Template
from sklearn.neighbors import KDTree
from scipy.interpolate import splrep, splev

#'''
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
#'''

f_test = 1
f_compile = 0

if f_compile:
    fp = open("/Users/jinyi/Work/PG_QSO/templates/DL07spec/dl07.tmplt", "r")
    tmpl_dl07 = pickle.load(fp)
    fp.close()

    uminList = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.80, 1.00, 1.20,
            1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 7.00, 8.00, 10.0, 12.0,
            15.0, 20.0, 25.0]
    umaxList = [1e3, 1e4, 1e5, 1e6]
    qpahList = [0.47, 1.12, 1.77, 2.50, 3.19, 3.90, 4.58, 0.75, 1.49, 2.37, 0.10]
    mdust2mh = [0.01, 0.01, 0.0101, 0.0102, 0.0102, 0.0103, 0.0104, 0.00343,
                0.00344, 0.00359, 0.00206]
    XList = []
    tckList = []
    counter = 0
    for umin in uminList:
        umaxList_exp = [umin] + umaxList
        for umax in umaxList_exp:
            for loop_qpah in range(len(qpahList)):
                qpah = qpahList[loop_qpah]
                mdmh = mdust2mh[loop_qpah]
                fltr_umin = tmpl_dl07["umin"] == umin
                fltr_umax = tmpl_dl07["umax"] == umax
                fltr_qpah = tmpl_dl07["qpah"] == qpah
                fltr = fltr_umin & fltr_umax & fltr_qpah
                wave = tmpl_dl07[fltr]["wavesim"][0]
                flux = tmpl_dl07[fltr]["fluxsim"][0]
                sortIndex = np.argsort(wave)
                wave = wave[sortIndex]
                flux = flux[sortIndex]
                tck = splrep(wave, flux)
                tckList.append(tck)
                XList.append([umin, umax, qpah])
                counter += 1
    kdt = KDTree(XList)
    print("Interpolation finishes!")
    wavelength = tmpl_dl07[0]["wavesim"]
    modelInfo = {
        "umin": uminList,
        "umax": umaxList,
        "qpah": qpahList,
        "mdmh": mdust2mh,
        "wavelength": wavelength,
    }

    parFormat = ["umin", "umax", "qpah"]
    readMe = '''
    This template is from: http://www.astro.princeton.edu/~draine/dust/irem.html
    The interpolation is tested well!
    '''
    templateDict = {
        "tckList": tckList,
        "kdTree": kdt,
        "parList": XList,
        "modelInfo": modelInfo,
        "parFormat": parFormat,
        "readMe": readMe
    }
    print("haha")
    t = Template(tckList=tckList, kdTree=kdt, parList=XList, modelInfo=modelInfo,
                 parFormat=parFormat, readMe=readMe)
    print("haha")
    t = Template(**templateDict)
    print("haha")

    fp = open("dl07_kdt.tmplt", "w")
    #pickle.dump(t, fp)
    pickle.dump(templateDict, fp)
    fp.close()

if f_test:
    fp = open("dl07_kdt.tmplt", "r")
    tpDict = pickle.load(fp)
    fp.close()
    fp = open("/Users/jinyi/Work/PG_QSO/templates/DL07spec/dl07.tmplt", "r")
    tmpl_dl07 = pickle.load(fp)
    fp.close()

    t = Template(**tpDict)
    x = 10**np.linspace(0, 3, 1000)
    pars = [0.44, 2e6, 2.3]
    for i in range(100): #range(len(tmpl_dl07)):
        umin = tmpl_dl07[i]["umin"]
        umax = tmpl_dl07[i]["umax"]
        qpah = tmpl_dl07[i]["qpah"]
        wave = tmpl_dl07[i]["wavesim"]
        flux = tmpl_dl07[i]["fluxsim"]
        pars = [umin, umax, qpah]
        flux_intp = t(wave, pars)
        print np.max(abs(flux-flux_intp))

    print(t.get_parFormat())
    print(t.readme())
    print t.get_nearestParameters(pars)
