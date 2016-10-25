import os
import gsf
import traceback
import numpy as np
from optparse import OptionParser
from astropy.table import Table

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

targetList = options.list
if targetList is None: #If the target list is not provided, only fit one target according to the config file.
    configName = args[0]
    gsf.gsf_fitter(configName)
else: #If the target list is provided, fit the targets one by one.
    configName = args[0]
    if len(args) == 2:
        sedPath    = args[1]
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
        #Check existing results
        if not options.refit:
            fileList = os.listdir(".")
            if "{0}_bestfit.txt".format(targname) in fileList:
                print("\n***{0} has been fitted!\n".format(targname))
                continue
        if options.usename:
            fileList = os.listdir(".")
            configTry = "config_{0}.py".format(targname)
            if configTry in fileList:
                configName = configTry
        sedFile = sedPath + sedname
        try:
            gsf.gsf_fitter(configName, targname, redshift, sedFile)
        except:
            print("\n---------------------------")
            print("***Fitting {0} is failed!".format(targname))
            traceback.print_exc()
            print("---------------------------")
