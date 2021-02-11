## active_learning_dd.py

This file contains utility functions for running a single iteration of a given configuration. 
In particular, the function:

```
get_next_batch(training_loader_params, 
\              unlabeled_loader_params,
               model_params,
               task_names,
               next_batch_selector_params,
               dissimilarity_memmap_filename)
```

is called with dictionary configurations of the training dataset loader, unlabeled dataset loader, machine learning model parameters, learning task name in the dataset, and the paremeters for the next batch selector (i.e. the strategy). 
It creates the corresponding python objects with the arguments given in the dictionaries. 

You can see this function in action in the runner scripts at `chtc_runners/` which make use of the configurations at `params_configs/`. 
So, this is just a helper function, and thus one can opt to write an iteration script that hard-codes the parameters and arguments.  

---
## database_loaders

This subdirectory contains dataset loader classes. 
The job of a loader is to manage the preparation, preprocessing, and loading of the dataset for use by the strategy as required. 
Currently, only `CSVLoader` has been implemented and tested. 

---
## utils

Contains utility/helper functions for evaluation, metrics, Taylor-Butina cluster computation, and dissimilarity calculation. 
These are words by various strategies. 

---
## models

Contains machine learning model codebase for use by strategies. 

---
## next_batch_selector

Contains main codebase for strategy implementation. 
The abstract base class of all strategies is `NBSBase`; all implemented strategy classes inherit this class. 
`NBSBase` contains functions for sampling randomly or dissimilarly from a given set of compounds. 
It also contains the crucial function `select_next_batch` that should be implemented by a strategy class which defines which compounds are selected in the current iteration. 

The file `cluster_based_selector.py` contains the code for the `ClusterBasedWCSelector` class. 

The file `instance_based_selector.py` contains the `InstanceBasedWCSelector` class which just inherits the `ClusterBasedWCSelector` and modifies it slightly. 

The file `mab_selector.py` contains the `MABSelector` class which implements a Upper-confidence-bound (UCB) style MAB solution; estimates reward as trade-off between exploit and explore term. 

See the [thesis document]() for details and descriptions on these strategies. 

---
# Summary of how it works

For example, the CBWS strategy asks for access to fingerprints and task labels of the training dataset. 
A training dataset CSVLoader would return a 2D numpy array for the compound fingerprints and a 1D numpy array for the task labels. 
The CSVLoader class also takes care of deduplication via the `idx_id_col_name`. 
CBWS would then train a machine learning model (i.e. random forest) that takes fingerprints as input and predicts the task label. 

An unlabeled dataset loader is also needed for when the strategy requests fingerprints. 
CBWS would then compute predictions on the unlabeled compound fingerprints. 
These predictions are used for computing cluster properties of activity and uncertainty. 
CBWS would then use the cluster properties to make decisions on which compounds to select for exploitation and exploration. 