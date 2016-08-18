import numpy as np
import fit_functions as sedff

SED_Model_Fit = sedff.SED_Model_Fit

def chisqFunc(param_dl07, chisq_list, funcKws):
    umin, umax, qpah = param_dl07
    fltr_umin = chisq_list['umin'] == umin
    fltr_umax = chisq_list['umax'] == umax
    fltr_qpah = chisq_list['qpah'] == qpah
    fltr = fltr_umin & fltr_umax & fltr_qpah
    if np.sum(fltr) > 1:
        raise ValueError('The chisq_list is wrong!')
    nTerm = np.argmax(fltr)
    funcKws['param_dl07'] = param_dl07
    results = SED_Model_Fit(**funcKws)
    chisqList[nTerm]['chisq'] = results['ChiSQ']
    return None

def parallelFunc(thread, run_range, paramDL07_list, chisq_list, funcKws):
    listBgn, listEnd = run_range
    for loop in range(listBgn, listEnd):
        paramDL07 = paramDL07_list[loop]
        chisqFunc(paramDL07, chisq_list, funcKws)
    print 'Thread *{0}* finish!'.format(thread)
    return None
