import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from sedfit.fitter.template import Template
from scipy.interpolate import splrep, splev
from sklearn.neighbors import KDTree
from astropy.table import Table
from glob import glob

N0List = [5, 7.5, 10]
awList = [-0.50, -1.00, -1.50, -2.00, -2.50]
fwdList = [0.15, 0.30, 0.45, 0.60, 0.75]
thetawList = [30, 45]
thetasigList = [7.50, 10.00, 15.00]
aList = [-0.50, -1.00, -1.50, -2.00, -2.50, -3.00]
hList = [0.10, 0.20, 0.30, 0.40, 0.50]
tauV = 50
Rout = 500
iList = [0, 15, 30, 45, 60, 75, 90]
length = len(glob('template/CAT3D-WIND_SED_GRID/*.*')) * 7
print(length)
XList = []
tckList = []
counter = 0

for filename in glob('template/CAT3D-WIND_SED_GRID/*.*'):
	print(filename)
	f = Table.read(filename, format='ascii')
	if filename[36] != '1':
		N0 = int(filename[36])
		if N0 == 7:
			N0 = 7.5
		# print('N0 ', N0)
		fwd = float(filename[41:45])
		# print('fwd ', fwd)
		a = float(filename[47:52])
		# print('a ', a)
		h = float(filename[54:58])
		# print('h ', h)
		aw = float(filename[61:66])
		# print('aw ', aw)
		thetaw = float(filename[73:75])
		# print('thetaw ', thetaw)
		thetasig = float(filename[84:88])
		# print('thetasig ', thetasig)
	else:
		N0 = int(filename[36:38])
		# print('N0 ', N0)
		fwd = float(filename[42:46])
		# print('fwd ', fwd)
		a = float(filename[48:53])
		# print('a ', a)
		h = float(filename[55:59])
		# print('h ', h)
		aw = float(filename[62:67])
		# print('aw ', aw)
		thetaw = float(filename[74:76])
		# print('thetaw ', thetaw)
		thetasig = float(filename[85:89])
		# print('thetasig ', thetasig)
	for i in iList:
		# print(i)
		index = i/15 + 3
		flux = f['col{0}'.format(index)]/f['col1']
		tck = splrep(f['col2'], flux)
		tckList.append(tck)
		XList.append([a, h, N0, i, fwd, aw, thetaw, thetasig])
		print("[{0}%]".format(100. * (counter + 1) / length))
		counter += 1

print(counter)
wave = f['col2']
kdt = KDTree(XList)
print("Interpolation finishes!")
modelInfo = {
	"a": aList,
	"h": hList,
	"N0": N0List,
	"i": iList,
	'fwd': fwdList,
	'aw': awList,
	'thetaw': thetawList,
	'thetasig': thetasigList,
	"wavelength": wave,
}

parFormat = ["a", "h", "N0", "i", 'fwd', 'aw', 'thetaw', 'thetasig']
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

fp = open("template/Cat3d_H_wind.tmplt", "w")
# pickle.dump(t, fp)
pickle.dump(templateDict, fp)
fp.close()

# test of template

fp = open("template/Cat3d_H_wind.tmplt", "r")
tpDict = pickle.load(fp)
fp.close()
t = Template(**tpDict)

counter = 0
for filename in glob('template/CAT3D-WIND_SED_GRID/*.*'):
	print(filename)
	f = Table.read(filename, format='ascii')
	if filename[36] != '1':
		N0 = int(filename[36])
		if N0 == 7:
			N0 = 7.5
		# print('N0 ', N0)
		fwd = float(filename[41:45])
		# print('fwd ', fwd)
		a = float(filename[47:52])
		# print('a ', a)
		h = float(filename[54:58])
		# print('h ', h)
		aw = float(filename[61:66])
		# print('aw ', aw)
		thetaw = float(filename[73:75])
		# print('thetaw ', thetaw)
		thetasig = float(filename[84:88])
		# print('thetasig ', thetasig)
	else:
		N0 = int(filename[36:38])
		# print('N0 ', N0)
		fwd = float(filename[42:46])
		# print('fwd ', fwd)
		a = float(filename[48:53])
		# print('a ', a)
		h = float(filename[55:59])
		# print('h ', h)
		aw = float(filename[62:67])
		# print('aw ', aw)
		thetaw = float(filename[74:76])
		# print('thetaw ', thetaw)
		thetasig = float(filename[85:89])
		# print('thetasig ', thetasig)
	for i in iList:
		# print(i)
		index = i/15 + 3
		flux = f['col{0}'.format(index)]/f['col1']
		tck = splrep(f['col2'], flux)
		pars = [a, h, N0, i, fwd, aw, thetaw, thetasig]
		flux_intp = t(wave, pars)
		print(np.max(abs(flux - flux_intp) / flux_intp))
		print("[{0}%]".format(100. * (counter + 1) / length))
		counter += 1
	if counter > 1000:
		break

print(t.get_parFormat())
print(t.readme())
print(t.get_nearestParameters(pars))