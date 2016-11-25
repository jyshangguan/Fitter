import os
import gc
import sys
import gsf
import traceback
import importlib
import numpy as np
import rel_SED_Toolkit as sedt
from optparse import OptionParser
from astropy.table import Table

#Include the config directory#
#----------------------------#
if os.path.isdir("configs"):
    sys.path.append("configs/")

#Parse the commands#
#-------------------#
parser = OptionParser()
parser.add_option("-l", "--list", dest="list", default=None, metavar="FILE",
                  help="Provide a list of target info to fit.")
parser.add_option("-n", "--usename", dest="usename",
                  action="store_true", default=False,
                  help="Try to find the config file specified with the target name.")
parser.add_option("-r", "--refit", dest="refit",
                  action="store_true", default=False,
                  help="Refit the SED though there is a result found.")
(options, args) = parser.parse_args()
if len(args) == 0:
    raise AssertionError("The config file is not specified!")


configName = args[0]
config = importlib.import_module(configName.split(".")[0])
targetList = options.list
if targetList is None: #->If the target list is not provided, only fit one target according to the config file.
    targname = config.targname
    redshift = config.redshift
    sedFile  = config.sedFile
    sedPck = sedt.Load_SED(sedFile, config.sedRng, config.spcRng, config.spcRebin)
    with open(sedFile, "r") as f:
        linesT = f.readlines()
        exec linesT[-1][1:]
        config.parTruth = inputPars
        #print "parTruth:"
        #print config.parTruth
    gsf.fitter(targname, redshift, sedPck, config)
else: #->If the target list is provided, fit the targets one by one.
    if len(args) == 2:
        sedPath = args[1]
    else:
        sedPath = ""
    targTable = Table.read(targetList , format="ascii.ipac")
    nameList  = targTable["Name"]
    zList     = targTable["z"]
    sedList   = targTable["sed"]
    print("\n***There are {0} targets to fit!\n".format(len(nameList)))
    for loop in range(len(nameList)):
        targname = nameList[loop]
        redshift = zList[loop]
        sedname  = sedList[loop]
        if not options.refit: #Omit the target if there is a fitting result.
            fileList = os.listdir(".")
            if "{0}_bestfit.txt".format(targname) in fileList:
                print("\n***{0} has been fitted!\n".format(targname))
                continue
        if options.usename: #Try to use the config file of the target itself.
            fileList = os.listdir(".")
            configTry = "config_{0}.py".format(targname)
            if configTry in fileList:
                configName = configTry
        sedFile = sedPath + sedname
        sedPck = sedt.Load_SED(sedFile, config.sedRng, config.spcRng, config.spcRebin)
        with open(sedFile, "r") as f:
            linesT = f.readlines()
            exec linesT[-1][1:]
            config.parTruth = inputPars
            #print "parTruth:"
            #print config.parTruth
        try:
            gsf.fitter(targname, redshift, sedPck, config)
        except:
            print("\n---------------------------")
            print("***Fitting {0} is failed!".format(targname))
            traceback.print_exc()
            print("---------------------------")
        gc.collect()
