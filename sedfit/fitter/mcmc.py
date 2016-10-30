import acor
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import truncnorm
from time import time

from .. import fit_functions as sedff

#The log_likelihood function
lnlike = sedff.logLFunc
#The log_likelihood function using Gaussian process regression
lnlike_gp = sedff.logLFunc_gp

def lnprior(params, data, model, ModelUnct, unctDict=None):
    """
    Calculate the ln prior probability.
    """
    lnprior = 0.0
    parIndex = 0
    parDict = model.get_modelParDict()
    for modelName in model._modelList:
        parFitDict = parDict[modelName]
        for parName in parFitDict.keys():
            if parFitDict[parName]["vary"]:
                parValue = params[parIndex]
                parIndex += 1
                pr1, pr2 = parFitDict[parName]["range"]
                if (parValue < pr1) or (parValue > pr2):
                    lnprior -= np.inf
            else:
                pass
    #If the model uncertainty is concerned.
    if ModelUnct:
        lnf, lna, lntau =  params[parIndex:]
        #lnf, lntau =  params[parIndex:]
        if (lnf < unctDict["lnf"][0]) or (lnf > unctDict["lnf"][1]):
            lnprior -= np.inf
        if (lna < unctDict["lna"][0]) or (lna > unctDict["lna"][1]):
            lnprior -= np.inf
        if (lntau < unctDict["lntau"][0]) or (lntau > unctDict["lntau"][1]):
            lnprior -= np.inf
    return lnprior

def lnprob(params, data, model, ModelUnct, unctDict):
    """
    Calculate the probability at the parameter spacial position.
    """
    lp = lnprior(params, data, model, ModelUnct, unctDict)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params, data, model)

def lnprob_gp(params, data, model, ModelUnct, unctDict):
    """
    Calculate the probability at the parameter spacial position.
    The likelihood function consider the Gaussian process regression.
    """
    lp = lnprior(params, data, model, ModelUnct, unctDict)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_gp(params, data, model)

class EmceeModel(object):
    """
    The MCMC model for emcee.
    """
    def __init__(self, data, model, ModelUnct=False, unctDict=None, sampler=None):
        self.__data = data
        self.__model = model
        self.__modelunct = ModelUnct
        self.__unctDict = unctDict
        self.__sampler = sampler
        print("[EmceeModel]: {0}".format(sampler))
        if ModelUnct:
            self.__dim = len(model.get_parVaryList()) + 3
            #self.__dim = len(model.get_parVaryList()) + 2
            print("[EmceeModel]: ModelUnct is on!")
        else:
            self.__dim = len(model.get_parVaryList())
            print "[EmceeModel]: ModelUnct is off!"

    def from_prior(self):
        """
        The prior of all the parameters are uniform.
        """
        parList = []
        parDict = self.__model.get_modelParDict()
        unctDict = self.__unctDict
        for modelName in self.__model._modelList:
            parFitDict = parDict[modelName]
            for parName in parFitDict.keys():
                if parFitDict[parName]["vary"]:
                    parRange = parFitDict[parName]["range"]
                    parType  = parFitDict[parName]["type"]
                    if parType == "c":
                        #print "[DN4M]: continual"
                        r1, r2 = parRange
                        p = (r2 - r1) * np.random.rand() + r1 #Uniform distribution
                    elif parType == "d":
                        #print "[DN4M]: discrete"
                        p = np.random.choice(parRange, 1)[0]
                    else:
                        raise TypeError("The parameter type '{0}' is not recognised!".format(parType))
                    parList.append(p)
                else:
                    pass
        #If the model uncertainty is concerned.
        if self.__modelunct:
            if unctDict is None:
                raise Error("No uncertainty model parameter range is provided!")
            rf1, rf2 = unctDict["lnf"]
            ra1, ra2 = unctDict["lna"]
            rt1, rt2 = unctDict["lntau"]
            lnf =  (rf1 - rf2) * np.random.rand() + rf2
            lna =  (ra1 - ra2) * np.random.rand() + ra2
            lntau =  (rt1 - rt2) * np.random.rand() + rt2
            parList.append(lnf)
            parList.append(lna)
            parList.append(lntau)
        parList = np.array(parList)
        return parList

    def EnsembleSampler(self, nwalkers, **kwargs):
        if self.__modelunct:
            self.__lnprob = lnprob_gp
        else:
            self.__lnprob = lnprob
        self.sampler = emcee.EnsembleSampler(nwalkers, self.__dim, self.__lnprob,
                       args=[self.__data, self.__model, self.__modelunct, self.__unctDict],
                       **kwargs)
        self.__nwalkers = nwalkers
        self.__sampler = "EnsembleSampler"
        return self.sampler

    def PTSampler(self, ntemps, nwalkers, **kwargs):
        if self.__modelunct:
            self.__lnlike = lnlike_gp
        else:
            self.__lnlike = lnlike
        self.sampler = emcee.PTSampler(ntemps, nwalkers, self.__dim,
                       logl=self.__lnlike, logp=lnprior,
                       loglargs=[self.__data, self.__model],
                       logpargs=[self.__data, self.__model, self.__modelunct], **kwargs)
        self.__ntemps = ntemps
        self.__nwalkers = nwalkers
        self.__sampler = "PTSampler"
        return self.sampler

    def p_ball(self, p0, ratio=5e-2, nwalkers=None):
        """
        Generate the positions of parameters around the input positions.
        The scipy.stats.truncnorm is used to generate the truncated normal distrubution
        of the parameters within the prior ranges.
        """
        ndim = self.__dim
        if nwalkers is None:
            nwalkers = self.__nwalkers
        pRange = self.__model.get_parVaryRanges()
        unctDict = self.__unctDict
        if self.__modelunct:
            if unctDict is None:
                raise Error("No uncertainty model parameter range is provided!")
            pRange.append(unctDict["lnf"]) #For lnf
            pRange.append(unctDict["lna"])  #For lna
            pRange.append(unctDict["lntau"])  #For lntau
        pRange = np.array(pRange)
        sampler = self.__sampler
        if sampler == "EnsembleSampler":
            p = np.zeros((nwalkers, ndim))
            for d in range(ndim):
                r0, r1 = pRange[d]
                std = (r1 - r0) * ratio
                loc = p0[d]
                a = (r0 - loc) / std
                b = (r1 - loc) /std
                p[:, d] = truncnorm.rvs(a=a, b=b, loc=loc, scale=std, size=nwalkers)
        if sampler == "PTSampler":
            ntemps = self.__ntemps
            p = np.zeros((ntemps, nwalkers, ndim))
            for t in range(ntemps):
                for d in range(ndim):
                    r0, r1 = pRange[d]
                    std = (r1 - r0) * ratio
                    loc = p0[d]
                    a = (r0 - loc) / std
                    b = (r1 - loc) /std
                    p[t, :, d] = truncnorm.rvs(a=a, b=b, loc=loc, scale=std, size=nwalkers)
        return p

    def p_prior(self):
        """
        Generate the positions in the parameter space from the prior.
        For EnsembleSampler, the result p0 shape is (nwalkers, dim).
        For PTSampler, the result p0 shape is (ntemps, nwalker, dim).
        """
        sampler  = self.__sampler
        nwalkers = self.__nwalkers
        dim      = self.__dim
        if sampler == "EnsembleSampler":
            p0 = [self.from_prior() for i in range(nwalkers)]
        elif sampler == "PTSampler":
            ntemps = self.__ntemps
            p0 = np.zeros((ntemps, nwalkers, dim))
            for loop_t in range(ntemps):
                for loop_w in range(nwalkers):
                    p0[loop_t, loop_w, :] = self.from_prior()
        else:
            raise ValueError("The sampler '{0}' is unrecognised!".format(sampler))
        return p0

    def p_logl_max(self):
        """
        Find the position in the sampled parameter space that the likelihood is
        the highest.
        """
        sampler = self.__sampler
        if sampler == "EnsembleSampler":
            lnlike = self.sampler.lnprobability
        else:
            lnlike = self.sampler.lnlikelihood
        chain  = self.sampler.chain
        idx = lnlike.ravel().argmax()
        p   = chain.reshape(-1, self.__dim)[idx]
        return p

    def p_logl_min(self):
        """
        Find the position in the sampled parameter space that the likelihood is
        the lowest.
        """
        sampler = self.__sampler
        if sampler == "EnsembleSampler":
            lnlike = self.sampler.lnprobability
        else:
            lnlike = self.sampler.lnlikelihood
        chain  = self.sampler.chain
        idx = lnlike.ravel().argmin()
        p   = chain.reshape(-1, self.__dim)[idx]
        return p

    def get_logl(self, p):
        """
        Get the likelihood at the given position.
        """
        sampler = self.__sampler
        if sampler == "EnsembleSampler":
            return self.__lnprob(p, self.__data, self.__model, self.__modelunct,
                                 self.__unctDict)
        elif sampler == "PTSampler":
            return self.__lnlike(p, self.__data, self.__model)
        else:
            raise ValueError("'{0}' is not recognised!".format(sampler))

    def run_mcmc(self, pos, iterations, printFrac=1, quiet=False, **kwargs):
        """
        Run the MCMC chain.
        This function just wraps up the sampler.sample() so that there is output
        in the middle of the run.
        """
        sampler = self.__sampler
        if not quiet:
            print("MCMC ({0}) is running...".format(sampler))
            t0 = time()
        #Notice that the third parameters yielded by EnsembleSampler and PTSampler are different.
        for i, (pos, lnprob, logl) in enumerate(self.sampler.sample(pos, iterations=iterations, **kwargs)):
            if not (i + 1) % int(printFrac * iterations):
                if quiet:
                    pass
                else:
                    progress = 100. * (i + 1) / iterations
                    if sampler == "EnsembleSampler":
                        lnlike = lnprob
                    elif sampler == "PTSampler":
                        lnlike = logl.ravel()
                    idx = lnlike.argmax()
                    lmax = lnlike[idx]
                    lmin = lnlike.min()
                    pmax = pos.reshape((-1, self.__dim))[idx]
                    pname = self.__model.get_parVaryNames(latex=False)
                    print("-----------------------------")
                    print("[{0:<4.1f}%] lnL_max: {1:.3e}, lnL_min: {2:.3e}".format(progress, lmax, lmin))
                    for p, name in enumerate(pname):
                        print("{0:18s} {1:10.3e}".format(name, pmax[p]))
                    print( "**MCMC time elapsed: {0:.3f} min".format( (time()-t0)/60. ) )
        if not quiet:
            print("MCMC finishes!")
        return pos, lnprob, logl

    def integrated_time(self):
        """
        Estimate the integrated autocorrelation time of a time series.
        Since it seems there is something wrong with the sampler.integrated_time(),
        I have to calculated myself using acor package.
        """
        sampler = self.__sampler
        if sampler == "EnsembleSampler":
            chain = self.sampler.chain
        elif sampler == "PTSampler":
            chain = self.sampler.chain[0, ...]
        else:
            raise ValueError("{0} is an unrecognised sampler!".format(sampler))
        tauParList = []
        for npar in range(self.__dim):
            tauList = []
            for nwal in range(self.__nwalkers):
                pchain = chain[nwal, :, npar]
                try:
                    tau, mean, sigma = acor.acor(pchain)
                except:
                    tau = np.nan
                tauList.append(tau)
            tauParList.append(tauList)
        return tauParList

    def accfrac_mean(self):
        """
        Return the mean acceptance fraction of the sampler.
        """
        return np.mean(self.sampler.acceptance_fraction)

    def posterior_sample(self, burnin=0, select=False, fraction=25):
        """
        Return the samples merging from the chains of all the walkers.
        """
        sampler  = self.sampler
        nwalkers = self.__nwalkers
        if self.__sampler == "EnsembleSampler":
            chain = sampler.chain
        elif self.__sampler == "PTSampler":
            chain = np.squeeze(sampler.chain[0, ...])
        if select:
            lnprob = sampler.lnprobability[:, -1]
            lnpLim = np.percentile(lnprob, fraction)
            fltr = lnprob > lnpLim
            print("ps: {0}/{1} walkers are selected.".format(np.sum(fltr), nwalkers))
            samples = chain[fltr, burnin:, :].reshape((-1, self.__dim))
        else:
            samples = chain[:, burnin:, :].reshape((-1, self.__dim))
        return samples

    def p_uncertainty(self, low=1, center=50, high=99, burnin=50, ps=None, **kwargs):
        """
        Return the uncertainty of each parameter according to its posterior sample.
        """
        if ps is None:
            ps = self.posterior_sample(burnin=burnin, **kwargs)
        else:
            pass
        parRange = np.percentile(ps, [low, center, high], axis=0)
        return parRange

    def print_parameters(self, truths=None, low=1, center=50, high=99, **kwargs):
        """
        Print the ranges of the parameters according to their posterior samples
        and the values of the maximum a posterior (MAP).
        """
        nameList = self.__model.get_parVaryNames(latex=False)
        parRange = self.p_uncertainty(low, center, high, **kwargs)
        if self.__modelunct:
            nameList.append("lnf")
            nameList.append("lna")
            nameList.append("lntau")
        pMAP = self.p_logl_max()
        ttList = ["Name", "L ({0}%)".format(low),
                  "C ({0}%)".format(center),
                  "H ({0}%)".format(high), "MAP"]
        if not truths is None:
            ttList.append("Truth")
        tt = " ".join(["{0:12s}".format(i) for i in ttList])
        print("{:-<74}".format(""))
        print(tt)
        for d in range(self.__dim):
            plow = parRange[0, d]
            pcen = parRange[1, d]
            phgh = parRange[2, d]
            pmax = pMAP[d]
            name = nameList[d]
            if (pmax < plow) or (pmax > phgh):
                print("[MCMC Warning]: The best-fit '{0}' is not consistent with its posterior sample".format(name))
            pl = [plow, pcen, phgh]
            info = "{0:12s} {1[0]:<12.3e} {1[1]:<12.3e} {1[2]:<12.3e} {2:<12.3e}".format(name, pl, pmax)
            if truths is None:
                print(info)
            else:
                print(info+" {0:<12.3e}".format(truths[d]))
        p_logl_max = self.p_logl_max()
        print("lnL_max: {0:.3e}".format(self.get_logl(p_logl_max)))

    def Save_Samples(self, filename, **kwargs):
        """
        Save the posterior samples.
        """
        samples = self.posterior_sample(**kwargs)
        np.savetxt(filename, samples, delimiter=",")

    def Save_BestFit(self, filename, low=1, center=50, high=99, **kwargs):
        nameList = self.__model.get_parVaryNames(latex=False)
        parRange = self.p_uncertainty(low, center, high, **kwargs)
        if self.__modelunct:
            nameList.append("lnf")
            nameList.append("lna")
            nameList.append("lntau")
        pMAP = self.p_logl_max()
        ttList = ["Name", "L ({0}%)".format(low),
                  "C ({0}%)".format(center),
                  "H ({0}%)".format(high), "MAP"]
        tt = " ".join(["{0:12s}".format(i) for i in ttList])
        fp = open(filename, "w")
        fp.write(tt+"\n")
        for d in range(self.__dim):
            plow = parRange[0, d]
            pcen = parRange[1, d]
            phgh = parRange[2, d]
            pmax = pMAP[d]
            name = nameList[d]
            pl = [plow, pcen, phgh]
            info = "{0:12s} {1[0]:<12.3e} {1[1]:<12.3e} {1[2]:<12.3e} {2:<12.3e}".format(name, pl, pmax)
            fp.write(info+"\n")
        p_logl_max = self.p_logl_max()
        fp.write("#lnL_max: {0:.3e}".format(self.get_logl(p_logl_max)))

    def plot_corner(self, filename=None, burnin=0, select=True, fraction=25, ps=None, nuisance=True, **kwargs):
        """
        Plot the corner diagram that illustrate the posterior probability distribution
        of each parameter.
        """
        if ps is None:
            ps = self.posterior_sample(burnin, select, fraction)
        parname = self.__model.get_parVaryNames()
        if self.__modelunct:
            parname.append(r"$\mathrm{ln}f$")
            parname.append(r"$\mathrm{ln}a$")
            parname.append(r"$\mathrm{ln}\tau$")
            nNui = 3 #The number of nuisance parameters
        else:
            nNui = 0
        if nuisance:
            dim = self.__dim
        else:
            dim = self.__dim - nNui
        fig = corner.corner(ps[:, 0:dim], labels=parname[0:dim], **kwargs)
        if filename is None:
            return fig
        else:
            plt.savefig(filename)
            plt.close()

    def plot_fit(self, filename=None, truths=None, FigAx=None, **kwargs):
        """
        Plot the best-fit model and the data.
        """
        sedData   = self.__data
        sedModel  = self.__model
        parRange  = self.p_uncertainty(**kwargs)
        waveModel = sedModel.get_xList()
        plow = parRange[0, :]
        pcnt = parRange[1, :]
        phgh = parRange[2, :]
        #pmax = self.p_logl_max()
        sedModel.updateParList(pcnt)
        ycnt = sedModel.combineResult() #The best-fit model
        ycnt_cmp = sedModel.componentResult() #The best-fit components
        yPhtC = np.array(sedData.model_pht(waveModel, ycnt)) #The best-fit band average flux density
        sedModel.updateParList(phgh)
        yhgh = sedModel.combineResult() #The upperlimit model
        yhgh_cmp = sedModel.componentResult() #The upperlimit components
        yPhtH = np.array(sedData.model_pht(waveModel, yhgh)) #The upperlimit band average flux density
        sedModel.updateParList(plow)
        ylow = sedModel.combineResult() #The lowerlimit model
        ylow_cmp = sedModel.componentResult() #The lowerlimit components
        yPhtL = np.array(sedData.model_pht(waveModel, ylow)) #The lowerlimit band average flux density
        #Plot the SED data
        fig, ax = sedData.plot_sed(FigAx=FigAx)
        cList = ["r", "g", "b", "m", "y", "c"]
        ncolor = len(cList)
        #Plot the best-fit model
        ax.plot(waveModel, ycnt, color="brown", linewidth=3.0, linestyle="--", label="Total")
        #Plot the upper- and lower- boundary
        ax.fill_between(waveModel, ylow, yhgh, color="brown", alpha=0.3)
        #Plot the band average photometric data with uncertainties
        ax.errorbar(sedData.get_dsList("x"), yPhtC, yerr=[yPhtC-yPhtL, yPhtH-yPhtC], marker="s",
                    color="r", mfc="none", mec="r", linestyle="none", label="Model")
        #Plot the different components of the model
        modelList = sedModel._modelList
        counter = 0
        for modelName in modelList:
            ax.plot(waveModel, ycnt_cmp[modelName], color=cList[counter%ncolor],
                    linestyle="--", label=modelName)
            ax.fill_between(waveModel, ylow_cmp[modelName], yhgh_cmp[modelName],
                             color=cList[counter], alpha=0.1)
            counter += 1
        #Plot the truth model if provided
        if not truths is None:
            sedModel.updateParList(truths)
            ytrue = sedModel.combineResult()
            ytrue_cmp = sedModel.componentResult()
            ax.plot(waveModel, ytrue, color="k", linestyle="-")
            counter = 0
            for modelName in modelList:
                ax.plot(waveModel, ytrue_cmp[modelName], color=cList[counter%ncolor])
                counter += 1
        #Setup the figure
        ax.set_xlabel(r"Wavelength ($\mu m$)", fontsize=24)
        ax.set_ylabel(r"$f_\nu$ (mJy)", fontsize=24)
        yData = sedData.get_List("y")
        ymin = 10**(np.floor(np.log10(min(yData))) - 2)
        ymax = 10**np.ceil(np.log10( max([max(yData), max(ycnt)]) ))
        ax.set_ylim([ymin, ymax])
        #Set the yaxis tick range
        yTickRange = ax.yaxis.get_majorticklocs()
        yTickRange = yTickRange[(yTickRange >= ymin) & (yTickRange <= ymax)]
        ax.yaxis.set_ticks(yTickRange[1:-1])
        ax.xaxis.set_tick_params(which="major", labelsize=18)
        ax.yaxis.set_tick_params(which="major", labelsize=18)
        ax.legend(loc="lower right", framealpha=0.3, fontsize=16)
        if filename is None:
            return (fig, ax)
        else:
            plt.savefig(filename, bbox_inches="tight")
            plt.close()

    def plot_fit_spec(self, filename=None, truths=None, FigAx=None, **kwargs):
        """
        Plot the best-fit model and the data.
        """
        sedData   = self.__data
        sedModel  = self.__model
        parRange  = self.p_uncertainty(**kwargs)
        waveModel = sedModel.get_xList()
        plow = parRange[0, :]
        pcnt = parRange[1, :]
        phgh = parRange[2, :]
        #pmax = self.p_logl_max()
        sedModel.updateParList(pcnt)
        ycnt = sedModel.combineResult()
        ycnt_cmp = sedModel.componentResult()
        sedModel.updateParList(phgh)
        yhgh = sedModel.combineResult()
        yhgh_cmp = sedModel.componentResult()
        sedModel.updateParList(plow)
        ylow = sedModel.combineResult()
        ylow_cmp = sedModel.componentResult()
        fig, ax = sedData.plot_sed(FigAx=FigAx)
        cList = ["r", "g", "b", "m", "y", "c"]
        ncolor = len(cList)
        ax.plot(waveModel, ycnt, color="brown", linewidth=1.5, linestyle="--", label="Total")
        ax.fill_between(waveModel, ylow, yhgh, color="brown", alpha=0.1)
        ax.set_xlabel("")
        ax.set_ylabel(r"$f_\nu$ (mJy)", fontsize=24)
        modelList = sedModel._modelList
        counter = 0
        for modelName in modelList:
            ax.plot(waveModel, ycnt_cmp[modelName], color=cList[counter%ncolor],
                    linestyle="--", label=modelName)
            ax.fill_between(waveModel, ylow_cmp[modelName], yhgh_cmp[modelName],
                             color=cList[counter], alpha=0.1)
            counter += 1
        if not truths is None:
            sedModel.updateParList(truths)
            ytrue = sedModel.combineResult()
            ytrue_cmp = sedModel.componentResult()
            ax.plot(waveModel, ytrue, color="k", linestyle="-")
            counter = 0
            for modelName in modelList:
                ax.plot(waveModel, ytrue_cmp[modelName], color=cList[counter%ncolor])
                counter += 1
        xData = sedData.get_csList("x")
        yData = sedData.get_csList("y")
        xmin = min(xData)*0.8
        xmax = max(xData)*1.2
        ymin = min(yData)*0.5
        ymax = max(yData)*2.0
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.xaxis.set_tick_params(which="major", labelsize=18)
        ax.yaxis.set_tick_params(which="major", labelsize=18)
        if filename is None:
            return (fig, ax)
        else:
            plt.savefig(filename, bbox_inches="tight")
            plt.close()

    def plot_chain(self, filename=None, truths=None):
        dim = self.__dim
        sampler = self.sampler
        nameList = self.__model.get_parVaryNames()
        if self.__modelunct:
            nameList.append(r"$\mathrm{ln}f$")
            nameList.append(r"$\mathrm{ln}a$")
            nameList.append(r"$\mathrm{ln}\tau$")
        if self.__sampler == "EnsembleSampler":
            chain = sampler.chain
        elif self.__sampler == "PTSampler":
            chain = np.squeeze(sampler.chain[0, ...])
        fig, axes = plt.subplots(dim, 1, sharex=True, figsize=(8, 3*dim))
        for loop in range(dim):
            axes[loop].plot(chain[:, :, loop].T, color="k", alpha=0.4)
            axes[loop].yaxis.set_major_locator(MaxNLocator(5))
            if not truths is None:
                axes[loop].axhline(truths[loop], color="r", lw=2)
            axes[loop].set_ylabel(nameList[loop], fontsize=24)
        if filename is None:
            return (fig, axes)
        else:
            plt.savefig(filename)
            plt.close()

    def plot_lnlike(self, filename=None, iterList=[0.5, 0.8, 1.0], **kwargs):
        lnprob = self.sampler.lnprobability
        _, niter = lnprob.shape
        iterList = np.around(niter * np.array(iterList)) - 1
        fig = plt.figure()
        for i in iterList:
            l = lnprob[:, int(i)]
            plt.hist(l[~np.isinf(l)], label="iter: {0}".format(i), **kwargs)
        plt.legend(loc="upper left")
        if filename is None:
            ax = plt.gca()
            return (fig, ax)
        else:
            plt.savefig(filename)
            plt.close()

    def reset(self):
        """
        Reset the sampler, for completeness.
        """
        self.sampler.reset()

    def diagnose(self):
        """
        Diagnose whether the MCMC run is reliable.
        """
        nameList = self.__model.get_parVaryNames(latex=False)
        if self.__modelunct:
            nameList.append("lnf")
            nameList.append("lna")
            nameList.append("lntau")
        print("---------------------------------")
        print("Mean acceptance fraction: {0:.3f}".format(self.accfrac_mean()))
        print("PN       : ACT (min-max)")
        it = self.integrated_time()
        for loop in range(self.__dim):
            itPar = it[loop]
            print("{0:9s}: {i[0]:.3f}-{i[1]:.3f}".format(nameList[loop], i=[min(itPar), max(itPar)]))

    def sampler_type(self):
        return self.__sampler

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict

    def __del__(self):
        del self.__data
        del self.__model
        del self.__modelunct
        del self.__unctDict
        del self.__sampler
        parList = self.__dict__.keys()
        if "sampler" in parList:
            del self.sampler
