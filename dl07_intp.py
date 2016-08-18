import numpy as np
from scipy.interpolate import interp1d
import cPickle as pickle

fp = open('/Users/jinyi/Work/PG_QSO/templates/DL07spec/dl07.tmplt', 'r')
tmpl_dl07 = pickle.load(fp)
fp.close()

uminList = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.80, 1.00, 1.20,
            1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 7.00, 8.00, 10.0, 12.0,
            15.0, 20.0, 25.0]
umaxList = [1e6]
qpahList = [0.47, 1.12, 1.77, 2.50, 3.19, 3.90, 4.58, 0.75, 1.49, 2.37, 0.10]
'''
uminList = [1.00, 10.0, 25.0]
umaxList = [1e6]
qpahList = [2.50, 4.58, 0.10]
'''

waveModel = 10**np.linspace(0, 3, 500)
nTemplate = len(uminList) * (len(umaxList) + 1) * len(qpahList)
nSEDpt_inpt = len(waveModel)
tmpl_dl07_inpt = np.empty(nTemplate, dtype=[('modelname', np.str_, 29), ('umin', np.float64),
                                            ('umax', np.float64), ('qpah', np.float64),
                                            ('wavesim', np.float64, (nSEDpt_inpt,)),
                                            ('fluxsim', np.float64, (nSEDpt_inpt,))])
counter = 0
for umin in uminList:
    umaxList_exp = [umin] + umaxList
    for umax in umaxList_exp:
        for qpah in qpahList:
            fltr_umin = tmpl_dl07['umin'] == umin
            fltr_umax = tmpl_dl07['umax'] == umax
            fltr_qpah = tmpl_dl07['qpah'] == qpah
            fltr = fltr_umin & fltr_umax & fltr_qpah
            wavesim = tmpl_dl07[fltr]['wavesim'][0]
            fluxsim = tmpl_dl07[fltr]['fluxsim'][0]
            fluxInpt = interp1d(wavesim, fluxsim)(waveModel)
            tmpl_dl07_inpt[counter]['modelname'] = tmpl_dl07[fltr]['modelname'][0]
            tmpl_dl07_inpt[counter]['umin'] = umin
            tmpl_dl07_inpt[counter]['umax'] = umax
            tmpl_dl07_inpt[counter]['qpah'] = qpah
            tmpl_dl07_inpt[counter]['wavesim'] = waveModel
            tmpl_dl07_inpt[counter]['fluxsim'] = fluxInpt
            counter += 1
print 'Finish DL07 model interpolation!'
fp = open('dl07_intp.dict', 'w')
pickle.dump(tmpl_dl07_inpt, fp)
fp.close()
