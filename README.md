# Active Learning in Drug Discovery

[![Build Status](https://travis-ci.com/gitter-lab/active-learning-drug-discovery.svg?token=DHS914p1QvszNqsQgFxW&branch=master)](https://travis-ci.com/gitter-lab/active-learning-drug-discovery)

## Citation

If you use this software please cite:

Moayad Alnammi, Spencer S. Ericksen, Scott A. Wildman, Nathan Wlodarchak, Hunter Reis, Troy King, Song Guo, Gene E. Ananiev, Anthony Gitter.
Iterative Batched Screening.
2021. [doi:xx.xxxx/xxxxxx]()

## Installation

We recommend creating a [conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) to manage the dependencies.
Assumes [Anaconda installation](https://www.anaconda.com/download/). 
Then, clone this repository:
```
git clone https://github.com/gitter-lab/active-learning-drug-discovery.git
cd active-learning-drug-discovery
```

The CISP informer set method uses the `aldd` conda environment using the `conda_env.yml` file:
```
conda env create -f conda_env.yml
source activate aldd
```

Finally, install `active-learning-drug-discovery` with `pip`.
```
pip install -e .
```

## datasets

The datasets used in this study are: PriA-SSB target, 107 PubChemBioAssay targets, and PstP target. 
The specific datasets used in this study can be downloaded from: [doi:xx.xxxx/zenodo.xxxxxxx](). 
The subdirectory also contains a small dataset for testing.

## active_learning_dd

The active_learning_dd subdirectory contains the main codebase for the iterative batched screening components. 
Consult the README in that subdirectory for details. 

## param_configs

This subdirectory contains json config files for strategies and experiments used in the [thesis document]().
Consult the README in that subdirectory for details. 

## analysis_notebooks

This subdirectory contains jupyter notebooks that preprocess the datasets, debug methods, analyze the results, and produce result images.

## runner scripts

`chtc_runners/` contains runner scripts for the experiments in the [thesis document]().
`chtc_runners/simulation_runner.py` can be used as a starting template for your own runner script. 
`chtc_runners/simulation_utils.py` contains helper functions for pre- and post-processing iteration selections for retrospective experiments. 
Consult the README in that subdirectory for details. 

## Implemented Iterative Strategies

The following are the currently implemented strategies in `active_learning_dd/next_batch_selector/` (see [thesis document]() and hyperapameter examples in `param_configs/`):

1. ClusterBasedWeightSelector (CBWS): assigns exploitation-exploration weights to every cluster, splits the budget between exploit-explore, then select compounds from most exploitable clusters, followed by selecting most explorable clusters. 

2. ClusterBasedRandom: randomly samples a cluster, then randomly samples compounds from within clusters. 

3. InstanceBasedRandom: randomly samples compounds from the pool. 

4. ClusterBasedDissimilar: samples clusters dissimilarly according to a dissimilarity measure which is by default fingerprint based. 

5. InstanceBasedDissimilar: samples compounds dissimilarly from the pool. 

6. MABSelector: Upper-Confidence-Bound (UCB) style solution from Multi-Armed Bandits (MAB). 
Assigns every cluster an upper-bound estimate of the reward that is a combination of a exploitation term and an exploration term. 
Samples clusters with the highest rewards. 
