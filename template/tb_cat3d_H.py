import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from sedfit.fitter.template import Template
from scipy.interpolate import splrep, splev
from sklearn.neighbors import KDTree
from astropy.table import Table
from glob import glob

aList = [-0.25, -0.50, -0.75, -1.00, -1.25, -1.50, -1.75, -2.00, -2.25, -2.50]
hList = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
tauV = 50
Rout = 500
iList = [0, 15, 30, 45, 60, 75, 90]
length = len(glob('template/CAT3D_SED_GRID/*.*')) * 7

XList = []
tckList = []
counter = 0
for filename in glob('template/CAT3D_SED_GRID/*.*'):
	print(filename)
	f = Table.read(filename, format='ascii')
	if filename[31] != '1':
		N0 = int(filename[31])
		print(N0)
		a = float(filename[42:47])
		print(a)
		h = float(filename[49:53])
		print(h)
	else:
		N0 = int(filename[31:33])
		print(N0)
		a = float(filename[43:48])
		print(a)
		h = float(filename[50:54])
		print(h)
	for i in iList:
		print(i)
		index = i/15 + 3
		flux = f['col{0}'.format(index)]/f['col1']
		tck = splrep(f['col2'], flux)
		tckList.append(tck)
		XList.append([a, h, N0, i])
		print("[{0}%]".format(100. * (counter + 1) / length))
		counter += 1

wave = f['col2']
kdt = KDTree(XList)
print("Interpolation finishes!")
modelInfo = {
	"a": aList,
	"N0": [5, 7, 10],
	"h": hList,
	"i": iList,
	"wavelength": wave,
}

parFormat = ["a", "h", "N0", "i"]
readMe = '''
	This template is from: http://www.sungrazer.org/cat3d.html
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

fp = open("template/Cat3d_H.tmplt", "w")
# pickle.dump(t, fp)
pickle.dump(templateDict, fp)
fp.close()

# test of template

fp = open("template/Cat3d_H.tmplt", "r")
tpDict = pickle.load(fp)
fp.close()
t = Template(**tpDict)

counter = 0
for filename in glob('template/CAT3D_SED_GRID/*.*'):
	f = Table.read(filename, format='ascii')
	if filename[31] != '1':
		N0 = int(filename[31])
		print(N0)
		a = float(filename[42:47])
		print(a)
		h = float(filename[49:53])
		print(h)
	else:
		N0 = int(filename[31:33])
		print(N0)
		a = float(filename[43:48])
		print(a)
		h = float(filename[50:54])
		print(h)
	for i in iList:
		index = i/15 + 3
		flux = f['col{0}'.format(index)]/f['col1']
		pars = [a, h, N0, i]
		flux_intp = t(wave, pars)
		print(np.max(abs(flux - flux_intp) / flux_intp))
		print("[{0}%]".format(100. * (counter + 1) / length))
		counter += 1
	if counter > 100:
		break

print(t.get_parFormat())
print(t.readme())
print(t.get_nearestParameters(pars))