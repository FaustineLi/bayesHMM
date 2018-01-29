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
        '''Intializes HMM object '''
        
        self.alpha = np.ones(n_states)
        self.mu  = 0
        self.tau = 3
        self.scale = 1

        if type(prior) is dict:
            self.alpha = np.array(prior.get('alpha', self.alpha))
            self.mu = prior.get('mu', self.mu)
            self.tau = prior.get('tau', self.tau)
            self.scale = prior.get('scale', self.scale)

        if np.sum(self.alpha > 0) != n_states:
            raise ValueError('number of positive alpha parameters must match number of states')

        if self._is_not_numeric(self.mu) or self._is_not_numeric(self.tau) or self._is_not_numeric(self.scale):
            raise ValueError('Prior parameters mu, tau, and scale must be a number')

        self.n_states = n_states
        self.n_samps = n_samps
        self.n_burn = n_burn
        self.n_iters = self.n_samps + self.n_burn + 1
        self.seed = seed

        # intialize transition matrix and random starting points
        np.random.seed(self.seed)

        self.transition = np.random.dirichlet(alpha=self.alpha, size=self.n_states)
        self.mean_samp  = np.zeros(shape=(self.n_iters, self.n_states))
        self.var_samp   = np.zeros(shape=(self.n_iters, self.n_states))

        self.mean_samp[0] = np.random.normal(
                                loc=self.mu, 
                                scale=self.tau, 
                                size=self.n_states) 
        self.var_samp[0]  = 1 / np.random.gamma(
                                shape=self.scale, 
                                scale=self.scale, 
                                size=self.n_states)

    def _is_not_numeric(self, x):
        '''checks if x is not numeric'''
        return type(x) not in (float, int)

    def _burn(self, x):
        '''Get rid of burn-in samples'''
        return x[self.n_burn+1:-1]


    def _standardize(self, y):
        '''helper function to transform data before sampling'''
        self._y_mean = np.mean(y)
        self._y_var = np.var(y)
        return (y - self._y_mean) / self._y_var


    def _unstandardize(self, y, mean=0, var=1):
        '''helper function to transform data after sampling'''
        return (y[self.n_burn:-1] * self._y_var) + self._y_mean


    def _update_backwards(self, y, i):
        '''updates backwards messsages'''
        for j in range(len(y)-2, 0, -1):
            norm = scipy.stats.norm.pdf(
                        x=y[j+1], 
                        loc=self.mean_samp[i], 
                        scale=np.sqrt(self.var_samp[i]))
            b = np.sum(self._beta[j+1] * self.transition * norm, axis=1)
            self._beta[j] = b / np.sum(b)


    def _update_latent_states(self, y, i):
        '''samples new latent states'''
        for j in range(1, len(y)-1):
            norm = scipy.stats.norm.pdf(
                        x=y[j], 
                        loc=self.mean_samp[i], 
                        scale=np.sqrt(self.var_samp[i]))
            p = self._beta[j] * norm  * self.transition[self._states[j-1]]
            self._states[j] = np.random.choice(
                                range(self.n_states), 
                                size=1, 
                                p=p / np.sum(p), 
                                replace=True)


    def _update_emission_params(self, y, i):
        '''updates parameters of emission distributions'''
        for h in range(self.n_states):
                
            n = np.sum(self._states == h)
            if n > 0:
                # update mu
                tau_n = 1 / (1 / self.tau + n / self.var_samp[i,h])
                mu_n  = tau_n * ((self.mu / self.tau + n * np.mean(y[self._states == h])) / 
                        self.var_samp[i,h])
                self.mean_samp[i+1, h] = np.random.normal(loc=mu_n, scale=np.sqrt(tau_n)) 

                # update varience
                vn = self.scale + n
                sn = (1 / vn) * ((self.scale * self.tau + np.sum((y[self._states == h] - 
                     self.mean_samp[i+1,h])**2)))
                self.var_samp[i+1,h] = 1 / np.random.gamma(shape=vn, scale=sn, size=1)
            else:
                # otherwise sample from prior
                self.mean_samp[i+1,h]  = np.random.normal(loc=self.mu, scale=np.sqrt(self.tau))
                self.var_samp[i+1,h] = 1 / np.random.gamma(shape=self.scale, scale=self.scale)


    def _update_transitions(self, y):
        '''updates transition proabilities'''
        count = np.zeros(shape=(self.n_states, self.n_states))
        for j in range(1, len(y)-1):
            count[self._states[j], self._states[j+1]] += 1

        for h in range(self.n_states):
            self.transition[h] = np.random.dirichlet(alpha=self.alpha + count[h])


    def _sample_predict(self, i):
        '''samples next time point'''
        self.x_pred[i] = np.random.choice(
                            range(self.n_states), 
                            p=self.transition[self._states[-1]])
        self.y_pred[i] = np.random.normal(
                            loc=self.mean_samp[i+1, self.x_pred[i]], 
                            scale=np.sqrt(self.var_samp[i+1, self.x_pred[i]]))

    def fit(self, y):
        '''
        Preforms inference for HMM using Gibbs sampling and the forward-backward algorithm
            y: array, 1-D input data
        '''
        np.random.seed(self.seed)

        self._beta = np.ones(shape=(len(y), self.n_states))
        self._states = np.random.choice(range(self.n_states), size=len(y), replace=True)
        self.y_pred = np.zeros(shape=self.n_iters)
        self.x_pred = np.zeros(shape=self.n_iters, dtype=int)
    
        # standardize data 
        y = self._standardize(y)

        for i in tqdm(range(self.n_iters-1)):

            self._update_backwards(y, i) 
            self._update_latent_states(y, i)
            self._update_emission_params(y, i)
            self._update_transitions(y)

            self._sample_predict(i)
        
        # throw away burn-in samples and unstandarize
        self.y_pred = self._burn(self._unstandardize(self.y_pred))
        self.x_pred = self._burn(self.x_pred)
        self.mean_samp = self._burn(self.mean_samp) + self._y_mean
        self.var_samp = self._burn(self.var_samp) * self._y_var