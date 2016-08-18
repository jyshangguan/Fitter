import numpy as np
import matplotlib.pyplot as plt
from gaussian_model import GaussFunc
import cPickle as pickle

fp = open("test_model.dict", "r")
model = pickle.load(fp)
fp.close()
ps = np.loadtxt("posterior_sample.txt")

xd = model['x']
yTrue = model['y_true']
yObsr = model['y_obsr']
yErr = model['y_err']
pValue = model['parameters']
cmpList = model['compnents']

nGauss = len(pValue)
for loop in range(nGauss):
    print "{0[0]}, {0[1]}, {0[2]}".format(pValue[loop])

#Calculate the optimized paramter values
parRangeList = map( lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                   zip(*np.percentile(ps, [16, 50, 84], axis=0)) )

#"""
par50List= []
par16List= []
par84List= []
print("Fitting results:")
for loop in range(nGauss):
    na = loop * 3 + 0
    nb = loop * 3 + 1
    nc = loop * 3 + 2
    prA = parRangeList[na]
    prB = parRangeList[nb]
    prC = parRangeList[nc]
    par50List.append( (prA[0], prB[0], prC[0]) )
    par84List.append( (prA[1], prB[1], prC[1]) )
    par16List.append( (prA[2], prB[2], prC[2]) )
    a_true, b_true, c_true = pValue[loop]
    print( "a_{0}: {1[0]}+{1[1]}-{1[2]} (True: {2})".format(loop, prA, a_true) )
    print( "b_{0}: {1[0]}+{1[1]}-{1[2]} (True: {2})".format(loop, prB, b_true) )
    print( "c_{0}: {1[0]}+{1[1]}-{1[2]} (True: {2})".format(loop, prC, c_true) )
print("-----------------")

fig = plt.figure()
plt.errorbar(xd, yObsr, yerr=yErr, fmt=".k")
plt.plot(xd, yTrue, linewidth=1.5, color="k")
for y in cmpList:
    plt.plot(xd, y, linestyle='--')

xm = np.linspace(1., 1000., 1000)
ym = np.zeros_like(xm)
for loop in range(nGauss):
    a50, b50, c50 = par50List[loop]
    y50 = GaussFunc(a50, b50, c50, xm)
    ym += y50
    a84, b84, c84 = par84List[loop]
    y84 = GaussFunc(a50+a84, b50+b84, c50+c84, xm)
    a16, b16, c16 = par16List[loop]
    y16 = GaussFunc(a50-a16, b50-b16, c50-c16, xm)
    plt.plot(xm, y50, color="r")
    plt.fill_between(xm, y16, y84, color="r", alpha=0.3)
plt.plot(xm, ym, color="r")
#plt.xlim([0, 100])
#plt.ylim([0, 800])
plt.show()
#"""
