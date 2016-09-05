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
f_compile = 1

if f_compile:
    h = h5py.File('/Users/jinyi/Work/PG_QSO/templates/clumpy_models_201410_tvavg.hdf5')
    h_q = h['q'].value
    h_N0 = h['N0'].value
    h_tv = h['tv'].value
    h_sig = h['sig'].value
    h_Y = h['Y'].value
    h_i = h['i'].value
    wave = h['wave'].value
    nu = ls_mic / wave
    kq   = np.sort(Counter(h_q).keys())
    kN0  = np.sort(Counter(h_N0).keys())
    ktv  = np.sort(Counter(h_tv).keys())
    ksig = np.sort(Counter(h_sig).keys())
    kY   = np.sort(Counter(h_Y).keys())
    ki   = np.sort(Counter(h_i).keys())

    XList = []
    tckList = []
    length = len(h_q)
    for counter in range(length):
        if (counter+1) % 100 == 0:
            print "[{0}%]".format(100. * (counter+1)/length)
        q   = h["q"][counter]
        N0  = h["N0"][counter]
        tv  = h["tv"][counter]
        sig = h["sig"][counter]
        Y   = h["Y"][counter]
        i   = h["i"][counter]
        flux = h["flux_tor"][counter]
        norm = np.trapz(flux, nu)
        flux_norm = flux/abs(norm)
        tck = splrep(wave, flux_norm)
        tckList.append(tck)
        XList.append([q, N0, tv, sig, Y, i])
    kdt = KDTree(XList)
    print("Interpolation finishes!")
    modelInfo = {
        "q": kq,
        "N0": kN0,
        "tv": ktv,
        "sigma": ksig,
        "Y": kY,
        "i": ki,
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
    h = h5py.File('/Users/jinyi/Work/PG_QSO/templates/clumpy_models_201410_tvavg.hdf5')

    wave = h["wave"].value
    nu = ls_mic / wave
    for counter in range(30): #range(len(tmpl_dl07)):
        q   = h["q"][counter]
        N0  = h["N0"][counter]
        tv  = h["tv"][counter]
        sig = h["sig"][counter]
        Y   = h["Y"][counter]
        i   = h["i"][counter]
        flux = h["flux_tor"][counter]
        norm = np.trapz(flux, nu)
        flux_norm = flux / abs(norm)
        pars = [q, N0, tv, sig, Y, i]
        flux_intp = t(wave, pars)
        print np.max(abs(flux_norm-flux_intp)/flux_intp)

    print(t.get_parFormat())
    print(t.readme())
    print t.get_nearestParameters(pars)
