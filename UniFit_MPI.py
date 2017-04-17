import os
import gc
import sys
import warnings
import traceback
import numpy as np
from optparse import OptionParser
from astropy.table import Table
from emcee.utils import MPIPool
import gsf_mpi as gsf

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
parser.add_option("-w", "--warning", dest="warning",
                  action="store_true", default=False,
                  help="Stop ignoring the warnings.")
(options, args) = parser.parse_args()
if len(args) == 0:
    raise AssertionError("The config file is not specified!")
#Some times the warning may stop the code, so we ignore the warnings by default.
if options.warning:
    pass
else:
    warnings.simplefilter("ignore")

pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

targetList = options.list
configName = args[0] #Get the input configure file information.
gsf.gsf_fitter(configName, pool)
