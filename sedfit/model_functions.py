import os
import numpy as np
from collections import OrderedDict
import cPickle as pickle
from dir_list import root_path
#-> Load the modelDict to select the modules to import
modelDictPath = "{0}temp_model.dict".format(root_path)
if os.path.isfile(modelDictPath):
    fp = open(modelDictPath, "r")
    modelDict = pickle.load(fp)
    fp.close()
    #--> No need to import all the modules
    import_all = 0
else:
    #--> Need to import all the modules
    import_all = 1
    print("Cannot find the temp_model.dict in {0}!".format(root_path))
#-> Load the modules
import_dict = {
    "model_bc03": ["BC03", "BC03_PosPar"],
    "model_bc03_refine": ["BC03_ref", "BC03_ref_PosPar"],
    "model_dl07": ["DL07", "DL07_PosPar"],
    "model_analyticals": ["Linear", "BlackBody", "Modified_BlackBody",
                          "Power_Law", "Synchrotron", "Line_Gaussian_L",
                          "Poly3"],
    "model_xl": ["Torus_Emission", "Torus_Emission_PosPar"],
    "model_torus_template": ["Torus_Template"],
    "model_pah": ["pah"],
    "model_cat3d_G": ["Cat3d_G", "Cat3d_G_PosPar"],
    "model_cat3d_H": ["Cat3d_H", "Cat3d_H_PosPar"],
    "model_cat3d_H_wind": ["Cat3d_H_wind", "Cat3d_H_wind_PosPar"],
    "model_extinction": ["Calzetti00"],
    "model_mir_extinction": ["Smith07"],
}
if import_all:
    for mds in import_dict.keys():
        funcList = import_dict[mds]
        exec "from models.{0} import {1}".format(mds, ",".join(funcList))
else:
    #-> Go through the functions in the modelDict
    for fnm in modelDict.keys():
        funcName = modelDict[fnm]["function"]
        #--> Go through the import_dict and find the modules in use
        for mds in import_dict.keys():
            funcList = import_dict[mds]
            if funcName in funcList:
                exec "from models.{0} import {1}".format(mds, ",".join(funcList))

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
    "Smith07": {
        "x_name": "wave",
        "param_fit": ["logtau"],
        "param_add": [],
        "operation": ["*"]
    },
    "Poly3": {
        "x_name": "x",
        "param_fit": ["c0", "c1", "c2", "c3"],
        "param_add": [],
        "operation": ["+"]
    }
}
