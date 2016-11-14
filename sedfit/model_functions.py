import numpy as np
from collections import OrderedDict
from model_bc03 import BC03
from model_clumpy import CLUMPY_intp
from model_dl07 import DL07
from model_analyticals import Linear, Modified_BlackBody, Power_Law, Line_Gaussian_L

"""
ls_mic = 2.99792458e14 #unit: micron/s
m_H = 1.6726219e-24 #unit: gram
Msun = 1.9891e33 #unit: gram
Mpc = 3.08567758e24 #unit: cm
mJy = 1e26 #unit: erg/s/cm^2/Hz
"""

#Dict of the supporting functions
funcLib = {
    "Linear":{
        "function": Linear,
        "x_name": "x",
        "param_fit": ["a", "b"],
        "param_add": []
    },
    "BC03":{
        "function": BC03,
        "x_name": "wave",
        "param_fit": ["logMs", "age"],
        "param_add": ["DL", "t"],
    },
    "CLUMPY_intp": {
        "function": CLUMPY_intp,
        "x_name": "wave",
        "param_fit": ["logL", "i", "tv", "q", "N0", "sigma", "Y"],
        "param_add": ["DL", "t"]
    },
    "DL07": {
        "function": DL07,
        "x_name": "wave",
        "param_fit": ["logumin", "logumax", "qpah", "gamma", "logMd"],
        "param_add": ["t", "DL"]
    },
    "Modified_BlackBody": {
        "function": Modified_BlackBody,
        "x_name": "wave",
        "param_fit": ["logM", "beta", "T"],
        "param_add": ["DL"]
    },
    "Power_Law": {
        "function": Power_Law,
        "x_name": "wave",
        "param_fit": ["PL_alpha", "PL_logsf"],
        "param_add": []
    },
    "Line_Gaussian_L": {
        "function": Line_Gaussian_L,
        "x_name": "wavelength",
        "param_fit": ["logLum", "lambda0", "FWHM"],
        "param_add": ["DL"]
    }
}
