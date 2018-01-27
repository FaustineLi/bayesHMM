import numpy as np
import scipy.stats
from tqdm import tqdm 

class HMM(object):
    '''
    Bayesian Hidden Markov Models

        n_states: int, number of latent states

        n_samps: int, number of sampling iterations

        n_burn: int, number of burn-in iterations

        prior: dict, set prior parameters

        seed: numeric, set random seed
    '''

    def __init__(self, n_states, n_samps, n_burn=0, prior=None, seed=None):

        if prior is None:
            self.alpha = np.ones(n_states)
            self.mu  = 0
            self.tau = 3
            self.scale = 1
        else:
            self.alpha = prior['alpha']
            self.mu = prior['mu']
            self.tau = prior['tau']
            self.scale = prior['scale']

        self.n_states = n_states
        self.n_samps = n_samps
        self.n_burn = n_burn
        self.n_iters = self.n_samps + self.n_burn + 1
        self.seed = seed

        # intialize transition matrix and random starting points
        np.random.seed(self.seed)

        self.transition = np.random.dirichlet(self.alpha, self.n_states)
        self.mean_samp  = np.zeros((self.n_iters, self.n_states))
        self.var_samp   = np.zeros((self.n_iters, self.n_states))

        self.mean_samp[0] = np.random.normal(self.mu, self.tau, self.n_states) 
        self.var_samp[0]  = 1 / np.random.gamma(self.scale, self.scale, self.n_states)

    def _standardize(self, y):
        '''helper function to transform data before sampling'''
        self._y_mean = np.mean(y)
        self._y_var = np.var(y)
        return (y - self._y_mean) / self._y_var

    def _unstandardize(self, y, mean=0, var=1):
        '''helper function to transform data after sampling'''

        return (y[self.n_burn:-1] * self._y_var) + self._y_mean

    def _burn(self, y):
        '''throw away burn in samples'''
        return y[self.n_burn:-1]


    def fit(self, y):

        '''
        Preforms inference for HMM using Gibbs sampling and the forward-backward algorithm

            y: array, 1-D input data

        '''

        np.random.seed(self.seed)

        self._beta = np.ones((len(y), self.n_states))
        self._states = np.random.choice(range(self.n_states), size=len(y), replace=True)
        self.y_pred = np.zeros(self.n_iters)
        self.x_pred = np.zeros(self.n_iters, dtype=int)
    
        # standarize data 
        y = self._standardize(y)

        for i in tqdm(range(self.n_iters-1)):

            # update backwards messages
            for j in range(len(y)-2, 0, -1):
                b = np.sum(
                        self._beta[j+1] * 
                        self.transition * 
                        scipy.stats.norm.pdf(y[j+1], self.mean_samp[i], np.sqrt(self.var_samp[i])),
                    axis=1)
                self._beta[j] = b / np.sum(b)

            # sample latent states
            for k in range(1, len(y)-1):
                p = self._beta[k] * \
                        scipy.stats.norm.pdf(y[k], self.mean_samp[i], np.sqrt(self.var_samp[i])) * \
                        self.transition[self._states[k-1]]
                self._states[k] = np.random.choice(range(self.n_states), size=1, p=p/np.sum(p), replace=True)


            # sample emission distributions
            for h in range(self.n_states):
                
                n = np.sum(self._states == h)
                if n > 0:
                    # update mu
                    tau_n = 1 / (1 / self.tau + n / self.var_samp[i,h])
                    mu_n  = tau_n * (self.mu / self.tau + n * np.mean(y[self._states == h])) / self.var_samp[i,h]
                    self.mean_samp[i+1, h] = np.random.normal(mu_n, np.sqrt(tau_n)) 

                    # update varience
                    vn = self.scale + n
                    sn = (1 / vn) * (self.scale * self.tau + np.sum((y[self._states == h] - self.mean_samp[i+1,h])**2))
                    self.var_samp[i+1,h] = 1 / np.random.gamma(vn, sn, 1)

                else:
                    # otherwise sample from prior
                    self.mean_samp[i+1,h]  = np.random.normal(self.mu, np.sqrt(self.tau))
                    self.var_samp[i+1,h] = 1 / np.random.gamma(self.scale, self.scale)

            # sample transition probabilities
            count = np.zeros((self.n_states, self.n_states))
            for k in range(1, len(y)-1):
                count[self._states[k], self._states[k+1]] += 1

            for h in range(self.n_states):
                self.transition[h] = np.random.dirichlet(self.alpha + count[h])

            # predict next time point
            self.x_pred[i] = np.random.choice(range(self.n_states), p=self.transition[self._states[-1]])
            self.y_pred[i] = np.random.normal(self.mean_samp[i+1, self.x_pred[i]], np.sqrt(self.var_samp[i+1, self.x_pred[i]]))
        
        # throw away burn-in samples and unstandarize
        self.y_pred = self._burn(self._unstandardize(self.y_pred))
        self.x_pred = self._burn(self.x_pred)
        self.mean_samp = self._burn(self.mean_samp) + self._y_mean
        self.var_samp = self._burn(self.var_samp) * self._y_var
