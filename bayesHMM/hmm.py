import numpy as np
import scipy.stats

class HMM:
    '''
    Bayesian Hidden Markov Model using Gibbs sampling and
    the forward / backward algorithm.  
    '''

    def __init__(self, n_states, n_samps=5000, n_burn = 0, prior=None, seed=None):

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
        self.mu_samp  = np.zeros((self.n_iters, self.n_states))
        self.var_samp  = np.zeros((self.n_iters, self.n_states))

        self.mu_samp[0] = np.random.normal(self.mu, self.tau, self.n_states) 
        self.var_samp[0] = 1 / np.random.gamma(self.scale, self.scale, self.n_states)

    def fit(self, X):

        self.beta = np.ones((len(X), self.n_states))
        self.beta[len(X)-1] = 1
        self.states = np.random.choice(range(self.n_states), size=len(X), replace=True)
        self.X_pred = np.zeros(self.n_iters)
        self.z_pred = np.zeros(self.n_iters, dtype=int)
    
        for i in range(self.n_iters-1):

            # update backwards messages
            for j in range(len(X)-2, 0, -1):
                b = np.sum(
                        self.beta[j+1] * 
                        self.transition * 
                        scipy.stats.norm.pdf(X[j+1], self.mu_samp[i], np.sqrt(self.var_samp[i])),
                    axis=1)
                self.beta[j] = b / np.sum(b)

            # sample latent states
            for k in range(1, len(X)-1):
                p = self.beta[k] * \
                        scipy.stats.norm.pdf(X[k], self.mu_samp[i], np.sqrt(self.var_samp[i])) * \
                        self.transition[self.states[k-1]]
                self.states[k] = np.random.choice(range(self.n_states), size=1, p=p/np.sum(p), replace=True)


            # sample emission distributions
            for h in range(self.n_states):
                
                n = np.sum(self.states == h)
                if n > 0:
                    # update mu
                    tau_n = 1 / (1 / self.tau + n / self.var_samp[i,h])
                    mu_n  = tau_n * (self.mu / self.tau + n * np.mean(X[self.states == h])) / self.var_samp[i,h]
                    self.mu_samp[i+1, h] = np.random.normal(mu_n, np.sqrt(tau_n)) 

                    # update varience
                    vn = self.scale + n
                    sn = (1 / vn) * (self.scale * self.tau + np.sum((X[self.states == h] - self.mu_samp[i+1,h])**2))
                    self.var_samp[i+1,h] = 1 / np.random.gamma(vn, sn, 1)
                else:
                    # otherwise sample from prior
                    self.mu_samp[i+1,h]  = np.random.normal(self.mu, np.sqrt(self.tau))
                    self.var_samp[i+1,h] = 1 / np.random.gamma(self.scale, self.scale)

            # sample transition probabilities
            count = np.zeros((self.n_states, self.n_states))
            for k in range(1, len(X)-1):
                count[self.states[k], self.states[k+1]] += 1

            for h in range(self.n_states):
                self.transition[h] = np.random.dirichlet(self.alpha + count[h])

            # predict next time point
            self.z_pred[i] = np.random.choice(range(self.n_states), p=self.transition[self.states[-1]])
            self.X_pred[i] = np.random.normal(self.mu_samp[i+1, self.z_pred[i]], np.sqrt(self.var_samp[i+1, self.z_pred[i]]))
        
        # throw away burn-in samples
        self.X_pred = self.X_pred[self.n_burn:self.n_iters-1]
        self.z_pred = self.z_pred[self.n_burn:self.n_iters-1]
        self.mu_samp = self.mu_samp[self.n_burn:self.n_iters-1]
        self.var_samp = self.var_samp[self.n_burn:self.n_iters-1]