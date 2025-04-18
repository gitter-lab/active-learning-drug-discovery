# Active Learning in Drug Discovery

[![Test active learning in drug discovery](https://github.com/gitter-lab/active-learning-drug-discovery/actions/workflows/test.yml/badge.svg)](https://github.com/gitter-lab/active-learning-drug-discovery/actions/workflows/test.yml)

<!--
## Citation

If you use this software please cite:

Moayad Alnammi, Spencer S. Ericksen, Scott A. Wildman, Nathan Wlodarchak, Hunter Reis, Troy King, Song Guo, Gene E. Ananiev, Anthony Gitter.
Iterative Batched Screening.
2021. [doi:xx.xxxx/xxxxxx]()
-->

## Installation

We recommend creating a [conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) to manage the dependencies.
Assumes [Anaconda installation](https://www.anaconda.com/download/). 
Clone this repository:
```
git clone https://github.com/gitter-lab/active-learning-drug-discovery.git
cd active-learning-drug-discovery
```

Setup the `active_learning_dd` conda environment using the `conda_cpu_env.yml` file:
```
conda env create -f conda_cpu_env.yml
conda activate active_learning_dd
```
If you do want GPU support, you can replace `conda_cpu_env.yml` with `conda_env.yml`.

Finally, install `active_learning_dd` with `pip`:
```
pip install -e .
```

Now check the installation is working correctly by running the sample data test:
```
cd chtc_runners
python sample_data_runner.py \
        --pipeline_params_json_file=../param_configs/sample_data_config.json \
        --hyperparams_json_file=../param_configs/experiment_PstP_hyperparams/sampled_hyparams/ClusterBasedWCSelector_609.json \
        --iter_max=5 \
        --no-precompute_dissimilarity_matrix \
        --initial_dataset_file=../datasets/sample_data/training_data/iter_0.csv.gz
 ```
 
 You should see the following last prompt:
 ```
 Finished testing sample dataset. Verified that hashed selection matches stored hash.
 ``` 

## datasets

The datasets used in this study are: PriA-SSB target, 107 PubChem BioAssay targets, and PstP target. 
<!--The specific datasets used in this study can be downloaded from: [doi:xx.xxxx/zenodo.xxxxxxx](). -->
The datasets will be uploaded to Zenodo in the near future.

The repository also contains a small dataset for testing: `datasets/sample_data/`. 

## active_learning_dd

The `active_learning_dd` subdirectory contains the main codebase for the iterative batched screening components. 
Consult the README in that subdirectory for details. 

## param_configs

This subdirectory contains json config files for strategies and experiments used in the [thesis document](https://www.biostat.wisc.edu/~gitter/pubs/AlnammiThesis.pdf).
Consult the README in that subdirectory for details. 

## analysis_notebooks

This subdirectory contains Jupyter notebooks that preprocess the datasets, debug methods, analyze the results, and produce result images.

## runner scripts

`chtc_runners/` contains runner scripts for the experiments in the [thesis document](https://www.biostat.wisc.edu/~gitter/pubs/AlnammiThesis.pdf).
`chtc_runners/simulation_runner.py` can be used as a starting template for your own runner script. 
`chtc_runners/simulation_utils.py` contains helper functions for pre- and post-processing iteration selections for retrospective experiments. 
Consult the README in that subdirectory for details. 

## Implemented Iterative Strategies

The following are the currently implemented strategies in `active_learning_dd/next_batch_selector/` (see [thesis document](https://www.biostat.wisc.edu/~gitter/pubs/AlnammiThesis.pdf) and hyperapameter examples in `param_configs/`):

1. **ClusterBasedWeightSelector (CBWS)**: assigns exploitation-exploration weights to every cluster, splits the budget between exploit-explore, then select compounds from most exploitable clusters, followed by selecting most explorable clusters. 

2. **ClusterBasedRandom**: randomly samples a cluster, then randomly samples compounds from within clusters. 

3. **InstanceBasedRandom**: randomly samples compounds from the pool. 

4. **ClusterBasedDissimilar**: samples clusters dissimilarly according to a dissimilarity measure which is by default fingerprint based. 

5. **InstanceBasedDissimilar**: samples compounds dissimilarly from the pool. 

6. **MABSelector**: Upper-Confidence-Bound (UCB) style solution from Multi-Armed Bandits (MAB). 
Assigns every cluster an upper-bound estimate of the reward that is a combination of a exploitation term and an exploration term. 
Samples clusters with the highest rewards. 
