import numpy as np
import matplotlib.pyplot as plt
from gaussian_model import MultiGaussian, GaussianModelDiscrete

#Generate the mock data#
#-----------------#
Ndata  = 50
xMax   = 1000.0
Nmodel = 1
fAdd   = None #0.1
pRange = [
    [5.0, 20.0],   #The range of a
    [20.0, 580.0], #The range of b
    [10.0, 100.0], #The range of c
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

fileName = "gauss{0}.dict".format(Nmodel)
fp = open(fileName, "w")
pickle.dump(model, fp)
fp.close()
print("{0} is saved!".format(fileName))

fig = plt.figure()
plt.errorbar(xd, yObsr, yerr=yErr, fmt=".k")
plt.plot(xd, yTrue, linewidth=1.5, color="k")
for y in cmpList:
    plt.plot(xd, y, linestyle='--')
plt.savefig("gauss{0}.pdf".format(Nmodel))
plt.close()
