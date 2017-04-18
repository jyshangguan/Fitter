from __future__ import print_function
import os
import sys
import warnings
import traceback
import numpy as np
from optparse import OptionParser
from astropy.table import Table

def makeCommand(cDict):
    """
    Make up the command line from the dict.
    """
    commandList = [cDict["head"]]
    for item in cDict["options"]:
        commandList.append(item)
    for item in cDict["args"]:
        commandList.append(item)
    cLine = " ".join(commandList)
    return cLine

#Parse the commands#
#-------------------#
parser = OptionParser()
parser.add_option("-l", "--list", dest="list", default=None, #metavar="FILE",
                  help="Provide a list of target info to fit.")
parser.add_option("-m", "--mpi_ncore", dest="ncores", default="1",
                  help="Run the code with MPI using the ")
parser.add_option("-r", "--refit", dest="refit",
                  action="store_true", default=False,
                  help="Refit the SED though there is a result found.")
parser.add_option("-o", "--overwrite", dest="overwrite",
                  action="store_true", default=False,
                  help="Overwrite the object information with the command-line inputs.")
parser.add_option("-w", "--warning", dest="warning",
                  action="store_true", default=False,
                  help="Stop ignoring the warnings.")
(options, args) = parser.parse_args()

commandDict = {}
ncores = eval(options.ncores)
if ncores == 1:
    commandHead = "python gsf.py"
elif ncores > 1:
    commandHead = "mpirun -np {0} python gsf_mpi.py".format(ncores)
commandDict = {
    "head": commandHead,
    #"options": [],
    #"args": args,
}
#print(commandLine)
targetList = options.list
if targetList is None:
    commandDict["options"] = []
    commandDict["args"] = args
    if options.overwrite:
        commandDict["options"].append("-o")
    if options.warning:
        commandDict["options"].append("-w")
    commandLine = makeCommand(commandDict)
    print(commandLine)
    os.system(commandLine)
