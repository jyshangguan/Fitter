import os
import sys

ncores = sys.argv[1]
pyFile = sys.argv[2]
os.system("mpirun -np {0} python {1}".format(ncores, pyFile))
