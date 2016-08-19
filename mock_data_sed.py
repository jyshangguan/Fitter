import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from gaussian_model import MultiGaussian, GaussianModelDiscrete

#Generate the mock data#
#-----------------#
Ndata  = 50
xMax   = 800.0
Nmodel = 10
fAdd   = None #0.1
pRange = [
    [5.0, 20.0],   #The range of a
    [20.0, 580.0], #The range of b
    [50.0, 200.0], #The range of c
]
print( "#------Start fitting------#" )
print( "# Ndata: {0}".format(Ndata) )
print( "# Nmodel: {0}".format(Nmodel) )
print( "# f_add: {0}".format(fAdd) )
print( "#-------------------------#" )

xData = np.linspace(1.0, xMax, Ndata)
model = MultiGaussian(xData, pRange, Nmodel, fAdd)
yTrue = model['y_true']
yObsr = model['y_obsr']
yErr = model['y_err']
pValue = model['parameters']
rangeList = model['ranges']
cmpList = model['compnents']
model['x'] = xData

for p in pValue:
    print("a: {0[0]}, b: {0[1]}, c:{0[2]}".format(p))

fileName = "gauss{0}.dict".format(Nmodel)
fp = open(fileName, "w")
pickle.dump(model, fp)
fp.close()
print("{0} is saved!".format(fileName))

fig = plt.figure()
plt.errorbar(xData, yObsr, yerr=yErr, fmt=".k")
plt.plot(xData, yTrue, linewidth=1.5, color="k")
for y in cmpList:
    plt.plot(xData, y, linestyle='--')
plt.savefig("gauss{0}.pdf".format(Nmodel))
plt.close()
