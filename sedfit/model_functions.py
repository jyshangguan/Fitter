import numpy as np
from collections import OrderedDict
from models.model_bc03 import BC03, BC03_PosPar
from models.model_bc03_refine import BC03_ref, BC03_ref_PosPar
from models.model_dl07 import DL07, DL07_PosPar
import models.model_analyticals as ma
from models.model_xl import Torus_Emission, Torus_Emission_PosPar
#from models.model_clumpy import CLUMPY_intp
from models.model_torus_template import Torus_Template
from models.model_pah import pah
from models.model_cat3d_G import Cat3d_G, Cat3d_G_PosPar
from models.model_cat3d_H import Cat3d_H, Cat3d_H_PosPar
from models.model_cat3d_H_wind import Cat3d_H_wind, Cat3d_H_wind_PosPar
from models.model_extinction import Calzetti00
#CLUMPY_intp = None

Linear = ma.Linear
BlackBody = ma.BlackBody
Modified_BlackBody = ma.Modified_BlackBody
Power_Law = ma.Power_Law
Synchrotron = ma.Synchrotron
Line_Gaussian_L = ma.Line_Gaussian_L
Poly3 = ma.Poly3

#Dict of the supporting functions
funcLib = {
    "Linear":{
        "x_name": "x",
        "param_fit": ["a", "b"],
        "param_add": [],
        "operation": ["+","*"]
    },
    "BC03":{
        "x_name": "wave",
        "param_fit": ["logMs", "age"],
        "param_add": ["DL", "z", "frame", "t"],
    },
    "BC03_ref":{
        "x_name": "wave",
        "param_fit": ["logMs", "logAge", "sfh"],
        "param_add": ["DL", "z", "frame", "t"],
    },
    "CLUMPY_intp": {
        "x_name": "wave",
        "param_fit": ["logL", "i", "tv", "q", "N0", "sigma", "Y"],
        "param_add": ["DL", "z", "frame", "t"]
    },
    "Torus_Emission": {
        "x_name": "wave",
        "param_fit": ["typeSil", "size", "T1Sil", "T2Sil", "logM1Sil", "logM2Sil",
                      "typeGra", "T1Gra", "T2Gra", "R1G2S", "R2G2S"],
        "param_add": ["DL", "z", "frame", "TemplateSil", "TemplateGra"]
    },
    "DL07": {
        "x_name": "wave",
        "param_fit": ["logumin", "logumax", "qpah", "loggamma", "logMd"],
        "param_add": ["t", "DL", "z", "frame"]
    },
    "BlackBody": {
        "x_name": "wave",
        "param_fit": ["logOmega", "T"],
        "param_add": []
    },
    "Modified_BlackBody": {
        "x_name": "wave",
        "param_fit": ["logM", "beta", "T"],
        "param_add": ["DL", "z", "kappa0", "lambda0", "frame"]
    },
    "Power_Law": {
        "x_name": "wave",
        "param_fit": ["PL_alpha", "PL_logsf"],
        "param_add": []
    },
    "Synchrotron": {
        "x_name": "wave",
        "param_fit": ["Sn_alpha", "Sn_logsf"],
        "param_add": ["lognuc", "lognum"]
    },
    "Line_Gaussian_L": {
        "x_name": "wavelength",
        "param_fit": ["logLum", "lambda0", "FWHM"],
        "param_add": ["DL"],
        "operation": ["+", "*"]
    },
    "pah": {
        "x_name": "wave",
        "param_fit": ["logLpah"],
        "param_add": ["t", "DL", "z", "frame", "waveLim"]
    },
    "Torus_Template": {
        "x_name": "wave",
        "param_fit": ["logLtorus"],
        "param_add": ["DL", "z", "frame", "ttype", "waveLim"]
    },
    "Cat3d_G": {
        "x_name": "wave",
        "param_fit": ["a", "theta", "N0", "i", "logL"],
        "param_add": ["DL", "z", "frame", "t"],
        "operation": ["+"]
    },
    "Cat3d_H": {
        "x_name": "wave",
        "param_fit": ["a", "h", "N0", "i", "logL"],
        "param_add": ["DL", "z", "frame", "t"],
        "operation": ["+"]
    },
    "Cat3d_H_wind": {
        "x_name": "wave",
        "param_fit": ["a", "h", "N0", "i", 'fwd', 'aw', 'thetaw', 'thetasig', "logL"],
        "param_add": ["DL", "z", "frame", "t"],
        "operation": ["+"]
    },
    "Calzetti00": {
        "x_name": "wave",
        "param_fit": ["Av", "Rv"],
        "param_add": ["waveLim", "QuietMode"],
        "operation": ["*"]
    },
    "Poly3": {
        "x_name": "x",
        "param_fit": ["c0", "c1", "c2", "c3"],
        "param_add": [],
        "operation": ["+"]
    }
}
