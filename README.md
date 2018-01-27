# Bayesian Hidden Markov Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An implementation of Bayesian hidden Markov models (HMMs) in Python for the analysis of dynamic systems. Inference is done using Gibbs sampling and [foward-backward algorithm](https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm).  

## Installation

Dependencies required include:

* Python (3.6) 
* NumPy (>=1.12.1)
* SciPy (>=0.19)

In addition `tqdm` is required for a progress bar.

To install, clone this repository with the command:

    git clone https://github.com/FaustineLi/bayesHMM.git

Navigate to the cloned directory and run this command:
  
    python setup.py install

## Documentation 

    class HMM(object):
      '''Bayesian hidden markov models using Gibbs sampling'''
      def __init__(self, n_states, n_samps, n_burn = 0, prior=None, seed=None):


### Parameters

* **n_states**: *integer*
    - Number of hidden or latent states. Must be greater than 1. 
     
* **n_samps**: *integer*
    - Set a number of Gibbs sampling iterations. Does not include burn-in iterations
    
* **n_burn**: *integer, optional* 
    - Set a number of burn-in or warm-up iterations. Default is no burn-in. 

* **prior**: *dict, optional*
     - A dictionary of prior specifications for the model. Default is a uninformative prior. Mean and varience priors should be for standarized data. Keywords are:
        - **alpha**: parameters for the Dirichlet distribution; array length should match number of latent states. 
        - **mu**: mean value for normal prior on emission distributions.
        - **tau**: varience value for normal prior on emission distributions. 
        - **scale**: parameter for inverse gamma prior. 
 
 * **seed**: *integer, optional*
    - Set a random seed for reproducibility.
    
    
## Usage

The example code below fits the HMM with two latent states to the data `y` and returns the posterior predictive values for the next time step. 

    from bayesHMM import HMM
    model = HMM(n_states=2, n_samp=5000, n_burn=1000)
    model.fit(y)
    model.y_pred
