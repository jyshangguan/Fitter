import acor
import emcee
import corner
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

from .. import fit_functions as sedff
logLFunc = sedff.logLFunc #The log_likelihood function

def lnprior(params, data, model, ModelUnct):
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
        lnf =  params[parIndex]
        if (lnf < -20) or (lnf > 1.0):
            lnprior -= np.inf
    return lnprior

def log_likelihood(params, data, model):
    """
    Gaussian sampling distrubution.
    """
    logL  = logLFunc(params, data, model)
    return logL

def lnprob(params, data, model, ModelUnct):
    lp = lnprior(params, data, model, ModelUnct)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, data, model)

class EmceeModel(object):
    """
    The MCMC model for emcee.
    """
    def __init__(self, data, model, ModelUnct=False, sampler=None):
        self.__data = data
        self.__model = model
        self.__modelunct = ModelUnct
        self.__sampler = sampler
        self.__dim = len(model.get_parVaryList())
        print("[EmceeModel]: {0}".format(sampler))
        if ModelUnct:
            print("[EmceeModel]: ModelUnct is on!")
        else:
            print "[EmceeModel]: ModelUnct is off!"

    def from_prior(self):
        """
        The prior of all the parameters are uniform.
        """
        parList = []
        parDict = self.__model.get_modelParDict()
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
            lnf =  20.0 * np.random.rand() - 10.0
            parList.append(lnf)
        parList = np.array(parList)
        return parList

    def EnsembleSampler(self, nwalkers, **kwargs):
        self.sampler = emcee.EnsembleSampler(nwalkers, self.__dim, lnprob,
                       args=[self.__data, self.__model, self.__modelunct], **kwargs)
        self.__nwalkers = nwalkers
        self.__sampler = "EnsembleSampler"
        self.__lnprob = lnprob
        return self.sampler

    def PTSampler(self, ntemps, nwalkers, **kwargs):
        self.sampler = emcee.PTSampler(ntemps, nwalkers, self.__dim,
                       logl=log_likelihood, logp=lnprior,
                       loglargs=[self.__data, self.__model],
                       logpargs=[self.__data, self.__model, self.__modelunct], **kwargs)
        self.__ntemps = ntemps
        self.__nwalkers = nwalkers
        self.__sampler = "PTSampler"
        self.__lnlike = log_likelihood
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
        pRange = np.array(self.__model.get_parVaryRanges())
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

    def get_logl_max(self):
        """
        Get the maximum of the likelihood at current particle distribution.
        """
        pmax = self.p_logl_max()
        sampler = self.__sampler
        if sampler == "EnsembleSampler":
            return self.__lnprob(pmax, self.__data, self.__model, self.__modelunct)
        elif sampler == "PTSampler":
            return self.__lnlike(pmax, self.__data, self.__model)
        else:
            raise ValueError("'{0}' is not recognised!".format(sampler))

    def burn_in(self, pos, iterations, printFrac=1, quiet=False, **kwargs):
        """
        Burn in the MCMC chain.
        This function just wraps up the sampler.sample() so that there is output
        in the middle of the run.
        """
        if not quiet:
            print("MCMC ({0}) is burning-in...".format(self.__sampler))
        for i, (pos, lnprob, state) in enumerate(self.sampler.sample(pos, iterations=iterations, **kwargs)):
            if not i % int(printFrac * iterations):
                if quiet:
                    pass
                else:
                    print("{0}%".format(100. * i / iterations))
        if not quiet:
            print("Burn-in finishes!")
        return pos, lnprob, state

    def run_mcmc(self, pos, iterations, printFrac=1, quiet=False, **kwargs):
        """
        Run the MCMC chain.
        This function just wraps up the sampler.sample() so that there is output
        in the middle of the run.
        """
        sampler = self.__sampler
        if not quiet:
            print("MCMC ({0}) is running...".format(sampler))
        #Notice that the third parameters yielded by EnsembleSampler and PTSampler are different.
        for i, (pos, lnprob, logl) in enumerate(self.sampler.sample(pos, iterations=iterations, **kwargs)):
            if not i % int(printFrac * iterations):
                if quiet:
                    pass
                else:
                    progress = 100. * i / iterations
                    if sampler == "EnsembleSampler":
                        lnlike = lnprob
                    elif sampler == "PTSampler":
                        lnlike = logl.ravel()
                    idx = lnlike.argmax()
                    lmax = lnlike[idx]
                    pmax = pos.reshape((-1, self.__dim))[idx]
                    pname = self.__model.get_parVaryNames(latex=False)
                    print("-----------------------------")
                    print("[{0:.1f}%] logL_max = {1:.3e}".format(progress, lmax))
                    for i, name in enumerate(pname):
                        print("{0:18s} {1:10.3e}".format(name, pmax[i]))
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
        tauList = []
        for np in range(self.__dim):
            pchain = chain[:, :, np].mean(axis=0)
            tau, mean, sigma = acor.acor(pchain)
            tauList.append(tau)
        return tauList

    def accfrac_mean(self):
        """
        Return the mean acceptance fraction of the sampler.
        """
        return np.mean(self.sampler.acceptance_fraction)

    def posterior_sample(self, burnin=0):
        """
        Return the samples merging from the chains of all the walkers.
        """
        sampler = self.sampler
        if self.__sampler == "EnsembleSampler":
            chain = sampler.chain
        elif self.__sampler == "PTSampler":
            chain = np.squeeze(sampler.chain[0, ...])
        samples = chain[:, burnin:, :].reshape((-1, self.__dim))
        return samples

    def p_uncertainty(self, low=1, center=50, high=99, burnin=50):
        """
        Return the uncertainty of each parameter according to its posterior sample.
        """
        ps = self.posterior_sample(burnin)
        parRange = np.percentile(ps, [low, center, high], axis=0)
        return parRange

    def print_parameters(self, truths=None, **kwargs):
        """
        Print the ranges of the parameters according to their posterior samples
        and the values of the maximum a posterior (MAP).
        """
        nameList = self.__model.get_parVaryNames(latex=False)
        parRange = self.p_uncertainty(**kwargs)
        pMAP = self.p_logl_max()
        ttList = ["name", "low", "center", "high", "map"]
        if not truths is None:
            ttList.append("truth")
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

    def Save_Samples(self, filename, burnin=50):
        """
        Save the posterior samples.
        """
        samples = self.posterior_sample(burnin)
        np.savetxt(filename, samples, delimiter=",")

    def plot_corner(self, filename=None, truths=None, burnin=50):
        """
        Plot the corner diagram that illustrate the posterior probability distribution
        of each parameter.
        """
        ps = self.posterior_sample(burnin)
        parname = self.__model.get_parVaryNames()
        fig = corner.corner(ps, labels=parname, truths=truths)
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
        phgh = parRange[2, :]
        pmax = self.p_logl_max()
        sedModel.updateParList(pmax)
        ymax = sedModel.combineResult()
        ymax_cmp = sedModel.componentResult()
        sedModel.updateParList(phgh)
        yhgh = sedModel.combineResult()
        yhgh_cmp = sedModel.componentResult()
        sedModel.updateParList(plow)
        ylow = sedModel.combineResult()
        ylow_cmp = sedModel.componentResult()
        fig, ax = sedData.plot_sed(FigAx=FigAx)
        cList = ["r", "g", "b", "m", "y", "c"]
        ncolor = len(cList)
        ax.plot(waveModel, ymax, color="brown", linewidth=3.0)
        ax.fill_between(waveModel, ylow, yhgh, color="brown", alpha=0.1)
        modelList = sedModel._modelList
        counter = 0
        for modelName in modelList:
            ax.plot(waveModel, ymax_cmp[modelName], color=cList[counter%ncolor])
            ax.fill_between(waveModel, ylow_cmp[modelName], yhgh_cmp[modelName],
                             color=cList[counter], alpha=0.1)
            counter += 1
        if not truths is None:
            sedModel.updateParList(truths)
            ytrue = sedModel.combineResult()
            ytrue_cmp = sedModel.componentResult()
            ax.plot(waveModel, ytrue, color="k", linestyle="--")
            counter = 0
            for modelName in modelList:
                ax.plot(waveModel, ytrue_cmp[modelName], color=cList[counter%ncolor])
                counter += 1
        if filename is None:
            return (fig, ax)
        else:
            plt.savefig(filename, bbox_inches="tight")
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
        print("---------------------------------")
        print("Mean acceptance fraction: {0:.3f}".format(self.accfrac_mean()))
        print("PN: ACT")
        print('\n'.join('{l[0]}: {l[1]:.3f}'.format(l=k) for k in enumerate(self.integrated_time())))

    def sampler_type(self):
        return self.__sampler

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict
