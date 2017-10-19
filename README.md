# Fitter

## Requirements

[Anachonda](https://www.continuum.io/) is suggested to use. Current version of Fitter is only tested on python 2.7 verion.

The code requires the following python packages:
* [emcee](http://dan.iel.fm/emcee/current/): pip install emcee
* [acor](https://github.com/dfm/acor): pip install acor
* [corner](http://corner.readthedocs.io/en/latest/#): pip install corner
* [George](http://dan.iel.fm/george/current/): conda install -c conda-forge george
* [mpi4py](http://pythonhosted.org/mpi4py/): pip install mpi4py or conda install -c anaconda mpi4py

* Additional things
  * In order to install tmux under the user home directory, use the following code:
    ```
    export DIR="$HOME/Softwares"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DIR/lib
    ./configure --prefix=$DIR CFLAGS="-I$DIR/include" LDFLAGS="-L$DIR/lib"
    make
    make install
    ```
  * The line `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/Softwares/lib` should also be added into `~/.bashrc`.

## Installation
