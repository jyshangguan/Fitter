import os
import gsf
import numpy as np
from optparse import OptionParser
from astropy.table import Table

#Parse the commands#
#-------------------#
parser = OptionParser()
parser.add_option("-c", "--config", dest="config", default="None",
                  help="Provide the configure file.", metavar="FILE")
parser.add_option("-l", "--list", dest="list", default="None",
                  help="Provide a list of configure file.")
(options, args) = parser.parse_args()

#Load the input module#
#---------------------#
configName = options.config
listName   = options.list
if listName != "None":
    configList = np.loadtxt(listName, dtype="string")
    configType = 1
else:
    if configName == "None":
        raise ValueError("The config file is not specified!")
    configType = 0

#Target information
#targname = "PG0050+124"
#redshift = 0.061
targTable = Table.read("pg_info_sobt.ipac", format="ascii.ipac")
targList  = targTable["Name"]
zList     = targTable["z"]
sedPath = "/Users/jinyi/Work/PG_QSO/catalog/Data_SG/SEDs/"
fileList = os.listdir(".")
for loop in range(len(targList)):
    targname = targList[loop]
    redshift = zList[loop]
    #Check existing results
    if "{0}_bestfit.txt".format(targname) in fileList:
        print("\n***{0} has been fitted!\n".format(targname))
        continue
    if configType:
        configName = configList[loop]
    sedFile = sedPath+"{0}_rest.tsed".format(targname)
    gsf.gsf_run(targname, redshift, sedFile, configName)
