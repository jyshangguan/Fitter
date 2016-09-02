import acor
import emcee
import numpy as np
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
        return self.sampler

    def PTSampler(self, ntemps, nwalkers, **kwargs):
        self.sampler = emcee.PTSampler(ntemps, nwalkers, self.__dim,
                       logl=log_likelihood, logp=lnprior,
                       loglargs=[self.__data, self.__model],
                       logpargs=[self.__data, self.__model, self.__modelunct], **kwargs)
        self.__ntemps = ntemps
        self.__nwalkers = nwalkers
        self.__sampler = "PTSampler"
        return self.sampler

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

    def p_logl_max(self):
        """
        Find the position in the sampled parameter space that the likelihood is
        the highest.
        """
        lnprob = self.sampler.lnprobability
        chain  = self.sampler.chain
        idx = lnprob.ravel().argmax()
        p   = chain.reshape(-1, self.__dim)[idx]
        return p

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
        if not quiet:
            print("MCMC ({0}) is running...".format(self.__sampler))
        for i, (pos, lnprob, state) in enumerate(self.sampler.sample(pos, iterations=iterations, **kwargs)):
            if not i % int(printFrac * iterations):
                if quiet:
                    pass
                else:
                    print("{0}%".format(100. * i / iterations))
        if not quiet:
            print("MCMC finishes!")
        return pos, lnprob, state

    def reset(self):
        """
        Reset the sampler, for completeness.
        """
        self.sampler.reset()

    def diagnose(self):
        """
        Diagnose whether the MCMC run is reliable.
        """
        print("Mean acceptance fraction: {0:.3f}".format(self.accfrac_mean()))
        print("PN: ACT")
        print('\n'.join('{l[0]}: {l[1]:.3f}'.format(l=k) for k in enumerate(self.integrated_time())))

    def sampler_type(self):
        return self.__sampler

    def postProcess(self, filename):
        sampler = self.sampler
        if self.__sampler == "EnsembleSampler":
            chain = sampler.chain
        elif self.__sampler == "PTSampler":
            chain = np.squeeze(sampler.chain[0, ...])
        samples = chain.reshape((-1, self.__dim))
        np.savetxt(filename, samples, delimiter=",")

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict
