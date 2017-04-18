from __future__ import print_function
import os
import sys
import warnings
from optparse import OptionParser
from gsf_core import *

#Parse the commands#
#-------------------#
parser = OptionParser()
parser.add_option("-w", "--warning", dest="warning",
                  action="store_true", default=False,
                  help="Stop ignoring the warnings.")
parser.add_option("-o", "--overwrite", dest="overwrite",
                  action="store_true", default=False,
                  help="Overwrite the object information with the command-line inputs.")
(options, args) = parser.parse_args()
if len(args) == 0:
    raise AssertionError("The config file is not specified!")
else:
    configName = args[0] #Get the input configure file information.
#Some times the warning may stop the code, so we ignore the warnings by default.
if options.warning:
    pass
else:
    warnings.simplefilter("ignore")

#The starter of this module#
#--------------------------#
print("\n")
print("############################")
print("# Galaxy SED Fitter starts #")
print("############################")
print("\n")
#->The object can be provided by the configure file or be overwrite with the
#command-line inputs
len_args = len(args)
if not options.overwrite:
    if len_args > 1:
        print("**Warning[UniFit]: there are more arguments may not be used...")
    gsf_fitter(configName)
else:
    if len_args < 4:
        raise AssertionError("The object information is lacking!")
    if len_args == 4:
        targname = args[1]
        redshift = eval(args[2])
        distance = None
        sedFile  = args[3]
    elif len_args == 5:
        targname = args[1]
        redshift = eval(args[2])
        distance = eval(args[3]) #The distance should be in Mpc.
        sedFile  = args[4]
    else:
        print("**Warning[UniFit]: there are more arguments may not be used...")
    gsf_fitter(configName, targname, redshift, distance, sedFile)
