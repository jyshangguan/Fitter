__author__ = "Robert Nikutta <robert.nikutta@gmail.com>"
__version__ = '20150416'

import numpy as N
import warnings
from scipy import interpolate, ndimage

# Convert RuntimeWarnings, e.g. division by zero in some array elements, to Exceptions
warnings.simplefilter('error', RuntimeWarning)


class NdimInterpolation:

    """N-dimensional interpolation on data hypercubes.

    Operates on image(index) coordinates. Multi-linear or cubic-spline
    (default).

    """

    def __init__(self,data,theta,order=1,mode='log'):

        """Initialize an interpolator object.

        Parameters
        ----------
        data : n-dim array or 1-d array
            Model database to be interpolated. Sampled on a
            rectilinear grid (it need not be regular!). 'data' is
            either an n-dimensional array (hypercube), or a
            1-dimensional array. If hypercube, each axis corresponds
            to one of the model parameters, and the index location
            along each axis grows with the parameter value. The last
            axis is the 'wavelength' axis. If 'data' is a 1-d array of
            values, it will be converted into the hypercube
            format. This means that the order of entries in the 1-d
            array must be as if constructed via looping over all axes,
            i.e.

              counter = 0
              for j0 in theta[0]:
                  for j1 in theta[1]:
                     for j2 in theta[2]:
                         ...
                         hypercube[j0,j1,j2,...] = onedarray[counter]
                         counter += 1

        theta : list
            List of 1-d arrays, each holding in ascending order the
            unique values for one of the model parameters. The last
            1-d array in theta is the wavelength array. Example: for
            the CLUMPY models of AGN tori (Nenkova et al. 2008)

              theta = [{i}, {tv}, {q}, {N0}, {sig}, {Y}, {wave}]

            where the {.} are 1-d arrays of unique model parameter
            values, e.g.

              {i} = array([0,10,20,30,40,50,60,70,80,90]) (degrees).

        order : int
            Order of interpolation spline to be used. order=1
            (default) is multi-linear interpolation, order=3 is
            cubic-spline (quite a bit slower, and not necessarily
            better, especially for complicated n-dim functions. order=1
            is recommended.

        mode : str
            'log' is default, and will take log10(data) first, which
            severely improves the interpolation accuracy if the data
            span many orders of magnitude. This is of course only
            applicable if all entries in 'data' are greater than
            0. Any string other that 'log' will keep 'data' as-is.

        Returns
        -------
        NdimInterpolation instance.

        Example
        -------
        General way to use ndiminterpolation

          ipython --pylab
          In[0]: import ndiminterpolation as nd
          In[1]: ip, datacube, theta, mywave = nd.example()

        See also example.py in the same repository.
        """
        
        self.theta = theta   # list of lists of parameter values, unique, in correct order

        shape_ = [len(t) for t in self.theta]

        # determine if data is hypercube or list of 1d arrays
        if shape_ == data.shape:
            self.input = 'hypercube'
            self.data_hypercube = data
        else:
            self.input = 'linear'
            self.data_hypercube = data.reshape(shape_,order='F')

        # interpolation orders
        assert (order in (1,3)), "Interpolation spline order not supported! Must be 1 (linear) or 3 (cubic)."
        self.order = order

        # interpolate in log10 space?
        self.mode = mode
        if self.mode == 'log':
            try:
                self.data_hypercube = N.log10(self.data_hypercube)
            except RuntimeWarning:
                raise Exception, "For mode='log' all entries in 'data' must be > 0."

        # set up n 1-d linear interpolators for all n parameters in theta
        self.ips = []   # list of 1-d interpolator objects
        for t in self.theta:
            self.ips.append(interpolate.interp1d(t,N.linspace(0.,float(t.size-1.),t.size)))

        if self.order == 3:
            print "Evaluating cubic spline coefficients for subsequent use, please wait..."
            self.coeffs = ndimage.spline_filter(self.data_hypercube,order=3)
            print "Done."


    def get_coords(self,vector,pivots=None):
        """Take real-world vector of parameter values, return image coordinates.

        Parameters
        ----------

        vector : 1-d array
            Values of every parameter at which the interpolation
            should be performed. This is on the input parameter
            hypercube.

        pivots : 1-d array
            Pivot points on the output, on which to deliver the
            interpolation results.


        Examples
        --------

        vector = [0., 100.4, 2.4]
        pivots = [1.,3.]   # can be e.g. microns, or the running number of a photometric band, etc.

        Then, get_coords(vector,pivots) returns the image coordinates at
        parameter 1/2/3 = [0., 100.4, 2.4] and at pivot points = [1.,3.]

        """
        
        len_vector = len(vector)

        if pivots is None:
            pivots = self.theta[-1]   # pivots are theta without the wave/bands

        coords = N.zeros((len_vector+1,pivots.size))  # 'x-prime'

        for i,ip in enumerate(self.ips):
            if i < len_vector:
                # unfortunately we have to repeat most of the coordinate array,
                # because we want the interpolated output on multiple pivot points
                coords[i,:] = N.repeat( ip(vector[i]), pivots.size )
            else: 
                coords[i,:] = ip(pivots)

        return coords
            

    def __call__(self,vector,pivots):
        """Interpolate in N dimensions, using mapping to image coordinates."""

        if self.order == 1:
            aux = ndimage.map_coordinates(self.data_hypercube,self.get_coords(vector,pivots=pivots),order=1)
        elif self.order == 3:
            aux = ndimage.map_coordinates(self.coeffs,self.get_coords(vector,pivots=pivots),order=3,prefilter=False)

        if self.mode == 'log':
            aux = 10.**aux

        return aux
