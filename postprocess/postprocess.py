#!/Users/jinyi/anaconda/bin/python

from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib_version = eval(matplotlib.__version__.split(".")[0])
if matplotlib_version > 1:
    plt.style.use("classic")
plt.rc('font',family='Times New Roman')
import sys
import types
import numpy as np
import cPickle as pickle
import sedfit.SED_Toolkit as sedt
from sedfit.fitter import basicclass as bc
from sedfit.mcmc import mcmc_emcee as mcmc
from sedfit import sedclass as sedsc
from sedfit import model_functions as sedmf
from matplotlib.ticker import FuncFormatter, FormatStrFormatter

def mjrFormatter(x, pos):
    """
    Define the function to setup the major axis tick label.
    """
    return "$10^{{{0:.0f}}}$".format(np.log10(x))

def ticksFinder(ymin, ymax,
                yTicksTry=np.array([0, 1e0, 1e1, 1e2, 1e3, 1e4])):
    """
    Find the proper ticklabel from ymin to ymax in logscale.
    """
    yTicksLabels = yTicksTry[(yTicksTry>ymin) & (yTicksTry<ymax)]
    if len(yTicksLabels) > 1:
        midTick = (np.log10(ymax)+np.log10(ymin))/2.0
        fltr_label = np.argmin(np.abs(np.log10(yTicksLabels) - midTick))
        yTicksLabel = yTicksLabels[fltr_label]
    else:
        yTicksLabel = yTicksLabels[0]
    return yTicksLabel

#Parse the commands#
#-------------------#
fitrsFile = sys.argv[1]
fp = open(fitrsFile, "r")
fitrs = pickle.load(fp)
fp.close()

#The code starts#
#################
print("#################################")
print("# Galaxy SED Fitter postprocess #")
print("#################################")

################################################################################
#                                    Data                                      #
################################################################################
dataPck = fitrs["dataPck"]
targname = dataPck["targname"]
redshift = dataPck["redshift"]
distance = dataPck["distance"]
dataDict = dataPck["dataDict"]
sedPck = dataPck["sedPck"]
sedData = sedsc.setSedData(targname, redshift, distance, dataDict, sedPck, silent=True)


################################################################################
#                                   Model                                      #
################################################################################
modelPck = fitrs["modelPck"]
modelDict = modelPck["modelDict"]
print("The model info:")
parCounter = 0
for modelName in modelDict.keys():
    print("[{0}]".format(modelName))
    model = modelDict[modelName]
    for parName in model.keys():
        param = model[parName]
        if not isinstance(param, types.DictType):
            continue
        elif param["vary"]:
            print("-- {0}, {1}".format(parName, param["type"]))
            parCounter += 1
        else:
            pass
print("Varying parameter number: {0}".format(parCounter))
print("#--------------------------------#")

#Build up the model#
#------------------#
parAddDict_all = modelPck["parAddDict_all"]
funcLib    = sedmf.funcLib
waveModel = modelPck["waveModel"]
sedModel = bc.Model_Generator(modelDict, funcLib, waveModel, parAddDict_all)
parTruth = modelPck["parTruth"]   #Whether to provide the truth of the model
modelUnct = modelPck["modelUnct"] #Whether to consider the model uncertainty in the fitting
parAllList = sedModel.get_parVaryList()
if modelUnct:
    parAllList.append(-np.inf)
    parAllList.append(-np.inf)
    parAllList.append(-5)

#Build the emcee object#
#----------------------#
em = mcmc.EmceeModel(sedData, sedModel, modelUnct)

#posterior process settings#
#--------------------------#
ppDict   = fitrs["ppDict"]
psLow    = ppDict["low"]
psCenter = ppDict["center"]
psHigh   = ppDict["high"]
nuisance = ppDict["nuisance"]
fraction = 0
burnIn = 0
ps = fitrs["posterior_sample"]

#Plot the SED data and fit
sedwave = sedData.get_List("x")
sedflux = sedData.get_List("y")
spcwave = sedData.get_csList("x")
spcflux = sedData.get_csList("y")
if sedData.check_csData():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    #->Plot the upper panel
    xmin = np.nanmin(spcwave) * 0.9
    xmax = np.nanmax(spcwave) * 1.1
    ymin = np.nanmin(spcflux) * 0.8
    ymax = np.nanmax(spcflux) * 1.05
    xlim = [xmin, xmax]
    ylim = [ymin, ymax]
    em.plot_fit(truths=parTruth, FigAx=(fig, ax1), xlim=xlim, ylim=ylim, nSamples=100,
                burnin=burnIn, fraction=fraction, ps=ps, showLegend=False)
    #-->Set the labels
    xTickLabels = [10., 20.]
    ax1.set_xticks(xTickLabels)
    ax1.set_xticklabels(xTickLabels)
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    yTL = ticksFinder(ymin, ymax, yTicksTry=np.linspace(ymin, ymax, 20))
    yTickLabels = [np.around(yTL, decimals=-1*int(np.log10(yTL)))]
    ax1.set_yticks(yTickLabels)
    ax1.set_yticklabels(yTickLabels)
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.tick_params(axis="both", which="major", length=8, labelsize=18)
    ax1.tick_params(axis="both", which="minor", length=5)
    plotName = r"PG {0}${1}${2}".format(targname[2:6], targname[6], targname[7:])
    ax1.text(0.05, 0.8, "{0}".format(plotName),
             verticalalignment='bottom', horizontalalignment='left',
             transform=ax1.transAxes, fontsize=24,
             bbox=dict(facecolor='white', alpha=0.5, edgecolor="none"))
    #-->Set the legend
    phtName = dataDict["phtName"]
    spcName = dataDict["spcName"]
    handles, labels = ax1.get_legend_handles_labels()
    handleUse = []
    labelUse  = []
    for loop in range(len(labels)):
        lb = labels[loop]
        hd = handles[loop]
        if lb == "Hot_Dust":
            lb = "BB"
        #if lb == "CLUMPY":
        #    lb = "CLU"
        if lb == phtName:
            hd = hd[0]
        if lb != spcName:
            labelUse.append(lb)
            handleUse.append(hd)
        else:
            label_spc  = lb
            handle_spc = hd
    labelUse.append(label_spc)
    handleUse.append(handle_spc)
    ax1.legend(handleUse, labelUse, loc="lower right", ncol=2,
               framealpha=0.9, edgecolor="white", #frameon=False, #
               fontsize=16, labelspacing=0.3, columnspacing=0.5,
               handletextpad=0.3, numpoints=1, handlelength=(4./3.))
    #->Plot the lower panel
    xmin = np.min(sedwave) * 0.9
    xmax = np.max(sedwave) * 1.1
    ymin = np.min(sedflux) * 0.5
    ymax = np.max(sedflux) * 2.0
    xlim = [xmin, xmax]
    ylim = [ymin, ymax]
    em.plot_fit(truths=parTruth, FigAx=(fig, ax2), xlim=xlim, ylim=ylim, nSamples=100,
                burnin=burnIn, fraction=fraction, ps=ps, showLegend=False)
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.tick_params(axis="both", which="major", length=8, labelsize=18)
    ax2.tick_params(axis="both", which="minor", length=5)
    #-->Set the labels
    yTicksLabels = [ticksFinder(ymin, ymax)]
    ax2.set_yticks(yTicksLabels)
    ax2.set_yticklabels(yTicksLabels)
    ax2.yaxis.set_major_formatter(FuncFormatter(mjrFormatter))
    plt.tight_layout(pad=1.8)
    #->Setup the shared axis label.
    ax.set_xlabel(r"Rest Wavelength ($\mu$m)", fontsize=24)
    ax.set_ylabel(r"$f_\nu \mathrm{(mJy)}$", fontsize=24)
    ax.xaxis.set_label_coords(0.5, -0.05)
    ax.yaxis.set_label_coords(-0.06, 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both',       # changes apply to the x-axis
                   which='both',      # both major and minor ticks are affected
                   bottom='off',      # ticks along the bottom edge are off
                   top='off',         # ticks along the top edge are off
                   labelbottom='off', # labels along the bottom edge are off)
                   labelleft="off")
else:
    fig = plt.figure(figsize=(7, 7))
    ax = plt.gca()
    xmin = np.min(sedwave) * 0.9
    xmax = np.max(sedwave) * 1.1
    ymin = np.min(sedflux) * 0.5
    ymax = np.max(sedflux) * 2.0
    xlim = [xmin, xmax]
    ylim = [ymin, ymax]
    em.plot_fit(truths=parTruth, FigAx=(fig, ax), xlim=xlim, ylim=ylim, nSamples=100,
                burnin=burnIn, fraction=fraction, ps=ps)
    xticks = [1., 2., 4., 8., 16.]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    #plotName = r"PG {0}${1}${2}".format(targname[2:6], targname[6], targname[7:])
    plotName = targname
    ax.text(0.05, 0.95, "{0}".format(plotName),
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes, fontsize=24,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor="none"))
    #-->Set the legend
    phtName = dataDict["phtName"]
    spcName = dataDict["spcName"]
    handles, labels = ax.get_legend_handles_labels()
    handleUse = []
    labelUse  = []
    for loop in range(len(labels)):
        lb = labels[loop]
        hd = handles[loop]
        if lb == "Hot_Dust":
            lb = "BB"
        #if lb == "CLUMPY":
        #    lb = "CLU"
        if lb == phtName:
            hd = hd[0]
        labelUse.append(lb)
        handleUse.append(hd)
    plt.legend(handleUse, labelUse, loc="upper left", fontsize=18, numpoints=1,
               handletextpad=0.3, handlelength=(4./3.), bbox_to_anchor=(0.02,0.90))
plt.savefig("{0}_result.pdf".format(targname), bbox_inches="tight")
plt.close()
print("Best fit plot finished!")

"""
#Plot the corner diagram
em.plot_corner(filename="{0}_triangle.png".format(targname), burnin=burnIn, ps=ps,
               nuisance=nuisance, truths=parTruth,  fraction=fraction,
               quantiles=[psLow/100., psCenter/100., psHigh/100.], show_titles=True,
               title_kwargs={"fontsize": 20})
print("Triangle plot finished!")
"""
