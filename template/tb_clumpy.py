import h5py
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from sedfit.fitter.template import Template
from sklearn.neighbors import KDTree
from scipy.interpolate import splrep, splev
from collections import Counter
ls_mic = 2.99792458e14 #micron/s

f_test = 1
f_compile = 0

qList   = [0., 1., 2., 3.]
N0List  = [1., 4., 8., 12., 15.]
tvList  = [10., 40., 80., 160., 300.]
sigList = [15., 30., 45., 60., 70.]
YList   = [5., 10., 30., 60., 100.]
iList   = [0., 20., 40., 70., 90.]
length = len(qList) * len(N0List) * len(tvList) * len(sigList) * len(YList) * len(iList)

h = h5py.File('/Users/jinyi/Work/PG_QSO/templates/clumpy_models_201410_tvavg.hdf5')
h_q = h['q'].value
h_N0 = h['N0'].value
h_tv = h['tv'].value
h_sig = h['sig'].value
h_Y = h['Y'].value
h_i = h['i'].value
wave = h['wave'].value
flux_tor = h["flux_tor"].value
nu = ls_mic / wave

if f_compile:
    kq   = np.sort(Counter(h_q).keys())
    kN0  = np.sort(Counter(h_N0).keys())
    ktv  = np.sort(Counter(h_tv).keys())
    ksig = np.sort(Counter(h_sig).keys())
    kY   = np.sort(Counter(h_Y).keys())
    ki   = np.sort(Counter(h_i).keys())

    XList = []
    tckList = []
    counter = 0
    for q in qList:
        for N0 in N0List:
            for tv in tvList:
                for sig in sigList:
                    for Y in YList:
                        for i in iList:
                            if (counter+1) % 100 == 0:
                                print "[{0}%]".format(100. * (counter+1)/length)
                            counter += 1
                            f_q   = h_q == q
                            f_N0  = h_N0 == N0
                            f_tv  = h_tv == tv
                            f_sig = h_sig == sig
                            f_Y   = h_Y == Y
                            f_i   = h_i == i
                            fltr = f_q & f_N0 & f_tv & f_sig & f_Y & f_i
                            flux = flux_tor[fltr][0] / nu
                            norm = np.trapz(flux, nu)
                            flux_norm = flux/abs(norm)
                            tck = splrep(wave, flux_norm)
                            tckList.append(tck)
                            XList.append([q, N0, tv, sig, Y, i])
    kdt = KDTree(XList)
    print("Interpolation finishes!")
    modelInfo = {
        "q": qList,
        "N0": N0List,
        "tv": tvList,
        "sigma": sigList,
        "Y": YList,
        "i": iList,
        "wavelength": wave,
    }

    parFormat = ["q", "N0", "tv", "sigma", "Y", "i"]
    readMe = '''
    This template is from: http://www.pa.uky.edu/clumpy/
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

    fp = open("clumpy_kdt.tmplt", "w")
    #pickle.dump(t, fp)
    pickle.dump(templateDict, fp)
    fp.close()

if f_test:
    fp = open("clumpy_kdt.tmplt", "r")
    tpDict = pickle.load(fp)
    fp.close()
    t = Template(**tpDict)

    counter = 0
    for q in qList:
        for N0 in N0List:
            for tv in tvList:
                for sig in sigList:
                    for Y in YList:
                        for i in iList:
                            if (counter+1) % 100 == 0:
                                print "[{0}%]".format(100. * (counter+1)/length)
                            counter += 1
                            f_q   = h_q == q
                            f_N0  = h_N0 == N0
                            f_tv  = h_tv == tv
                            f_sig = h_sig == sig
                            f_Y   = h_Y == Y
                            f_i   = h_i == i
                            fltr = f_q & f_N0 & f_tv & f_sig & f_Y & f_i
                            flux = flux_tor[fltr][0] / nu
                            norm = np.trapz(flux, nu)
                            flux_norm = flux/abs(norm)
                            pars = [q, N0, tv, sig, Y, i]
                            flux_intp = t(wave, pars)
                            print np.max(abs(flux_norm-flux_intp)/flux_intp)
        if counter > 100:
            break
    print(t.get_parFormat())
    print(t.readme())
    print t.get_nearestParameters(pars)
