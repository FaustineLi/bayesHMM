import unittest
import numpy as np
from unittest import TestCase
from bayesHMM import HMM

NSAMP  = 1000
NBURN  = 500
NSTATES = 3
SEED = 42

class hmmInitTests(TestCase):
        
    def test_load_params(self):
        '''testing loading well formed parameters'''
        PRIOR = {
            'alpha': [1, 3, 1],
            'mu': 1.1,
            'tau': 2,
            'scale': 2.1
        } 
        hmm = HMM(NSTATES, NSAMP, NBURN, prior=PRIOR, seed=SEED)
        self.assertTrue(hmm.n_states == NSTATES)
        self.assertTrue(hmm.n_samps == NSAMP)
        self.assertTrue(hmm.n_burn == NBURN)
        self.assertTrue(hmm.seed == SEED)
        
        self.assertTrue(np.all(hmm.alpha == PRIOR['alpha']))
        self.assertTrue(hmm.mu == PRIOR['mu'])
        self.assertTrue(hmm.tau == PRIOR['tau'])
        self.assertTrue(hmm.scale == PRIOR['scale'])

    def test_partial_load(self):
        '''testing partial loading of prior parameters'''
        PRIOR = {
            'tau': 2,
            'scale': 4
        }

        hmm = HMM(NSTATES, NSAMP, prior=PRIOR)
        self.assertTrue(hmm.tau == PRIOR['tau'])
        self.assertTrue(hmm.scale == PRIOR['scale'])
        self.assertTrue(hmm.mu == 0)
        self.assertTrue(np.all(hmm.alpha == [1, 1, 1]))


    def test_bad_alpha(self):
        '''testing malformed variations of alpha'''
        BAD_PRIOR1 = {'alpha': [3, -1, 2]}
        BAD_PRIOR2 = {'alpha': [1, 2, 3, 4]}
        BAD_PRIOR3 = {'alpha': [0, 1, 2, 3]}

        with self.assertRaises(ValueError):
            HMM(NSTATES, NSAMP, prior=BAD_PRIOR1)
            HMM(NSTATES, NSAMP, prior=BAD_PRIOR2)
            HMM(NSTATES, NSAMP, prior=BAD_PRIOR3)

    def test_bad_prior(self):
        '''testing malformed variations of mu, tau, and scale'''
        BAD_PRIOR1 = {'mu': [1, 2, 3]}
        BAD_PRIOR2 = {'tau': -2}
        BAD_PRIOR3 = {'tau': [1, 2, 3]}
        BAD_PRIOR4 = {'scale': -1}
        BAD_PRIOR5 = {'scale': [1, 2, 3]}
        with self.assertRaises(ValueError):
            HMM(NSTATES, NSAMP, prior=BAD_PRIOR1)
            HMM(NSTATES, NSAMP, prior=BAD_PRIOR2)
            HMM(NSTATES, NSAMP, prior=BAD_PRIOR3)
            HMM(NSTATES, NSAMP, prior=BAD_PRIOR4)
            HMM(NSTATES, NSAMP, prior=BAD_PRIOR5)

    def test_init(self):
        '''test intialization of data storage'''
        hmm = HMM(NSTATES, NSAMP, NBURN, seed=SEED)
        self.assertTrue(hmm.transition.shape == (NSTATES, NSTATES))
        self.assertTrue(np.all(hmm.mean_samp[0] != 0))
        self.assertTrue(np.all(hmm.var_samp[0] != 0))
        self.assertTrue(hmm.mean_samp.shape == (NSAMP+NBURN+1, NSTATES))
        self.assertTrue(hmm.var_samp.shape == (NSAMP+NBURN+1, NSTATES))



