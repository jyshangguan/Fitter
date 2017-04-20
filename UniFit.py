from __future__ import print_function
import os
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

#->Parse the commands
parser = OptionParser()
parser.add_option("-m", "--mpi_ncore", dest="ncores", default="1",
                  help="Run the code with MPI using the asigned number of cores if ncores>1.")
parser.add_option("-w", "--warning", dest="warning", action="store_true", default=False,
                  help="Stop ignoring the warnings.")
parser.add_option("-o", "--overwrite", dest="overwrite", action="store_true", default=False,
                  help="Overwrite the object information with the command-line inputs.")
parser.add_option("-l", "--list", dest="list", default=None,
                  help="Provide a list of target info to fit.")
parser.add_option("-f", "--list_format", dest="list_format", default="ascii.ipac",
                  help="Provide a list of target info to fit.")
parser.add_option("-r", "--refit", dest="refit", action="store_true", default=False,
                  help="Refit the SED though there is a result found. Only works with -l assigned.")
parser.add_option("-c", "--config", dest="config", default=None,
                  help="Assign the config file for a list of objects.")
parser.add_option("-p", "--path", dest="path", default=None,
                  help="Assign the path of the SED data for a list of objects.")
(options, args) = parser.parse_args()
if options.list is None:
    if not options.config is None:
        parser.error("Option -c only works with -l.")
    if not options.path is None:
        parser.error("Option -p only works with -l.")

#->Determine whether use MPI or not.
commandDict = {}
ncores = eval(options.ncores)
if ncores == 1:
    commandHead = "python gsf.py"
elif ncores > 1:
    commandHead = "mpirun -np {0} python gsf_mpi.py".format(ncores)
commandDict = {
    "head": commandHead,
}

#->Determine whether to run a list of objects.
targetList = options.list
if targetList is None:
    commandDict["options"] = []
    commandDict["args"] = args
    if options.overwrite:
        commandDict["options"].append("-o")
    if options.warning:
        commandDict["options"].append("-w")
    commandLine = makeCommand(commandDict)
    os.system(commandLine)
else: #If the target list is provided, fit the targets one by one.
    sedPath = options.path
    if sedPath is None:
        sedPath = ""
    targTable = Table.read(targetList, format=options.list_format)
    nameList  = targTable["Name"]
    zList     = targTable["z"]
    sedList   = targTable["sed"]
    print("\n***There are {0} targets to fit!\n".format(len(nameList)))
    for loop in range(len(nameList)):
        targname = nameList[loop]
        redshift = zList[loop]
        if options.config is None:
            try:
                configName = targTable["config"][loop]
            except:
                raise RuntimeError("Fail to find the config file information.")
        else:
            configName = options.config
        commandArgs = [configName, targname, "{0}".format(redshift)]
        if "DL" in targTable.colnames:
            distance = targTable["DL"][loop]
            commandArgs.append(distance)
        else:
            distance = None
        sedFile = sedPath + sedList[loop]
        commandArgs.append(sedFile)
        if not options.refit: #Omit the target if there is a fitting result.
            fileList = os.listdir(".")
            if "{0}.fitrs".format(targname) in fileList:
                print("\n***{0} has been fitted!\n".format(targname))
                continue
        commandDict["options"] = ["-o"] #We need to overwrite the target info.
        if options.warning:
            commandDict["options"].append("-w")
        commandDict["args"] = commandArgs
        commandLine = makeCommand(commandDict)
        try:
            os.system(commandLine)
        except:
            print("***Fail to run:\n    '{0}'!".format(commandLine))
