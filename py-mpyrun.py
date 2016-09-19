import os
import sys

ncores = sys.argv[1]
pyFile = sys.argv[2]
command = "mpirun -np {0} python {1} -m".format(ncores, pyFile)
os.system(command)
