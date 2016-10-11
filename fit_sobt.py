import os
import gsf
import importlib
from optparse import OptionParser
from astropy.table import Table

#Parse the commands#
#-------------------#
parser = OptionParser()
parser.add_option("-c", "--config", dest="config", default="config.py",
                  help="Provide the configure file", metavar="FILE")
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")
(options, args) = parser.parse_args()

#Load the input module#
#---------------------#
configName = options.config
print("Config file: {0}".format(configName))
cfg = importlib.import_module(configName.split(".")[0])

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
    sedFile = sedPath+"{0}_rest.nsed".format(targname)
    gsf.gsf_run(targname, redshift, sedFile, cfg)
