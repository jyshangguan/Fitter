import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.interpolate import interp1d
import fitter.basicclass as bc

def GaussFunc(a, b, c, x):
    return a * np.exp( -0.5 * ( (x - b) / c )**2. )

def MultiGaussian(x, p_range, n_model, f_add=None, QuietMode=True):
    """
    Generate a model with the combination of a number of Gaussian models.
    a is randomly chosen in the entire range for all the components.
    b is separated in small ranges so that the previous model is always in
    the front of the next model. And the values are randomly chosen in the log
    space.
    c is randomly chosen in the log space of the range.

    Parameters
    ----------
    x : float array
        The x of the Gaussian function.
    p_range : list
        The list of ranges for the three parameters of GaussFunc().
        Format: [[ra1, ra2], [rb1, rb2], [rc1, rc2]]
    n_model : int
        The number of Gaussian models.
    f_add : float
        The additional uncertainty that is not included in the yerr.
    QuietMode : bool
        The toggle that does not print if true.

    Returns
    -------
    model_pkg : dict
        The package of model.
        x: float array
            The x of the model.
        y_true: float array
            The true model without uncertainty.
        y_obsr: float array
            The observed data with uncertainty.
        y_err: float array
            The error uncertainty of the data.
        parameters: list of tuples.
            The parameter list of (a, b, c)
        compnents: list of float arrays.
            The model y list.
        ranges: list of lists.
            The list of parameter ranges [p1, p2].
        f_add: float
            The fraction of y data that cannot explained by the model.

    Notes
    -----
    None.
    """
    x_max = np.max(x)
    n_data = len(x)
    b1, b2 = p_range[1] #The range of the parameters
    rangeB = (b2 - b1) * np.random.rand(n_model - 1) + b1
    rangeB = np.sort( np.append(rangeB, [b1, b2]) )
    rangeList = [p_range[0], rangeB, p_range[2]]
    parList = [] #Record the parameters
    cmpList = [] #Record the models
    y_true = np.zeros_like(x)
    if not QuietMode:
        print("----------------------")
        print("Set model parameters:")
    #Set the model parameters in the parList and save the model components
    #in the cmpList
    for loop in range(n_model):
        ra1, ra2 = rangeList[0]
        a = (ra2 - ra1) * np.random.rand() + ra1
        rb1, rb2 = rangeList[1][loop:(loop+2)]
        lb1, lb2 = [np.log(rb1), np.log(rb2)]
        b = np.exp( (lb2 - lb1) * np.random.rand() + lb1 )
        rc1, rc2 = rangeList[2]
        lc1, lc2 = [np.log(rc1), np.log(rc2)]
        c = np.exp( (lc2 - lc1) * np.random.rand() + lc1 )
        if not QuietMode:
            print("a_{0}: {1:.3f} in ({2}, {3})".format(loop, a, ra1, ra2))
            print("b_{0}: {1:.3f} in ({2}, {3})".format(loop, b, rb1, rb2))
            print("c_{0}: {1:.3f} in ({2}, {3})".format(loop, c, rc1, rc2))
        parList.append( (a, b, c) )
        y = GaussFunc(a, b, c, x)
        cmpList.append(y)
        y_true += y

    #Add errors
    yerr = 0.5 + 1.5 * np.random.rand(n_data)
    y_obsr = y_true.copy()
    if not f_add is None:
        y_obsr += np.abs(f_add * y_obsr) * np.random.randn(n_data)
    y_obsr += yerr * np.random.randn(n_data)
    model_pkg = {
        'x': x,
        'y_true': y_true,
        'y_obsr': y_obsr,
        'y_err': yerr,
        'parameters': parList,
        'compnents': cmpList,
        'ranges': rangeList,
        'f_add': f_add,
    }
    return model_pkg

def GaussianModelSet(p_value, range_list):
    """
    Setup a multi-gaussian model to be used in fitting.

    parameters
    ----------
    p_value : list of tuples
        The parameter values (a, b, c) forming a tuple stored in the list.
    range_list : list
        The range of parameters for each model component. Format: [range_a,
        range_b, range_c]. The range_* is a list containing the sections
        of the number of the models.

    Returns
    -------
    gaussModel : class ModelCombiner()
        The combined model of Gaussian functions.

    Notes
    -----
    None.
    """
    n_model = len(p_value)
    parNameList = ['a', 'b', 'c']
    modelNameRoot = 'G{0}'
    modelDict = OrderedDict()
    for loop_m in range(n_model):
        #For each parameter, add the information.
        parFitDict = OrderedDict()
        for loop_pn in range(3):
            if len(range_list[loop_pn]) > 2:
                rp1, rp2 = range_list[loop_pn][loop_m:(loop_m+2)]
            else:
                rp1, rp2 = range_list[loop_pn]
            parFitDict[parNameList[loop_pn]] = {
                'value': p_value[loop_m][loop_pn],
                'range': [rp1, rp2],
                'type': 'c', #The parameter type discrete or continual.
                'vary': True,
            }
        gauss = bc.ModelFunction(GaussFunc, 'x', parFitDict)
        modelDict[modelNameRoot.format(loop_m)] = gauss
    gaussModel = bc.ModelCombiner(modelDict)
    return gaussModel

def GaussDiscrete(a, b, c, x, tmplt):
    """
    The Gaussian function with discrete paramters using a template.
    """
    fltrA = tmplt["a"] == a
    fltrB = tmplt["b"] == b
    fltrC = tmplt["c"] == c
    fltr = fltrA & fltrB & fltrC
    if np.sum(fltr) == 0:
        raise ValueError("The parameters are not on the grids!")
    xt = tmplt[fltr][0]["x"]
    yt = tmplt[fltr][0]["y"]
    y = interp1d(xt, yt)(x)
    return y

def GaussianModelDiscrete(n_model, range_dict, tmplt):
    """
    Setup a multi-gaussian model to be used in fitting.

    parameters
    ----------
    range_dict : dict
        The dict of the range for each paramter.
    tmplt : ndarray
        The template of Gaussian functions.

    Returns
    -------
    gaussModel : class ModelCombiner()
        The combined model of Gaussian functions.

    Notes
    -----
    None.
    """
    parNameList = ['a', 'b', 'c']
    parValueList = [10.0, 200.0, 50.0]
    modelNameRoot = 'G{0}'
    modelDict = OrderedDict()
    for loop_m in range(n_model):
        #For each parameter, add the information.
        parFitDict = OrderedDict()
        for loop_pn in range(3):
            parName = parNameList[loop_pn]
            parFitDict[parName] = {
                'value': parValueList[loop_pn],
                'range': range_dict[parName],
                'type': 'd', #The parameter type discrete or continual.
                'vary': True,
            }
        parAddDict = {"tmplt": tmplt}
        #gauss = bc.ModelFunction(GaussDiscrete, 'x', parFitDict, parAddDict)
        gauss = bc.ModelFunction(GaussFunc, 'x', parFitDict)
        modelDict[modelNameRoot.format(loop_m)] = gauss
    gaussModel = bc.ModelCombiner(modelDict)
    return gaussModel

if __name__ == "__main__":
    import cPickle as pickle
    fp = open("gt.dict", "r")
    gt = pickle.load(fp)
    fp.close()

    a = 7.0
    b = 260.0
    c = 65.0
    x = np.linspace(1.0, 1000.0, 500)
    yt = GaussDiscrete(a, b, c, x, gt)
    ym = GaussFunc(a, b, c, x)
    plt.plot(x, yt, "k", linewidth=1.5)
    plt.plot(x, ym, ":r", linewidth=1.5)
    plt.show()
