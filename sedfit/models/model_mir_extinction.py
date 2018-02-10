# Add extinction function
import numpy as np
from scipy import interpolate
from ..dir_list import template_path

f = np.loadtxt(template_path+'tau_lambda_kemper_new.txt')
xaxis = f[:, 0]
yaxis = f[:, 1]
k = interpolate.interp1d(xaxis,yaxis,kind='cubic')

def Smith07(logtau, wave):
    """
    This function adopts the extinction curve from Smith et al. (2007), section
    4.1.8, and extrapolated to 1000 micron by Mingyang Zhuang.
    """
    tempy1 = []; tempy2 = []
    extin_x = []
    for each in wave:
        if each < xaxis[0]:
            tempy1.append(0)
        elif each > xaxis[-1]:
            tempy2.append(0)
        else:
            extin_x.append(each)

    final_y = k(extin_x)
    extinction_list = np.concatenate((tempy1,final_y))
    extinction_list = np.concatenate((extinction_list, tempy2))
    ratio = np.exp(-10**logtau*extinction_list)
    return ratio
