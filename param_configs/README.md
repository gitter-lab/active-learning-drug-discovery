# Experiment Configs

Files like `exp_0_pipeline_config.json` and `exp_3_pipeline_config.json` contain configurations for the experiments in the thesis. 
These configs are used by the runner scripts in `chtc_runners/`. 
As such, they are not required for using the `active_learning_dd/` codebase, but serve as utility/helper for the runner scripts. 

---
# Strategy Hyperparameter Configs

The subdirectory `custom_hyperparams/benchmarks/` contains configuration examples for random, diversity, and MAB strategies. 
These configs are parameter settings for the implemented strategies in `active_learning_dd/` and serve as utility/helpers in runner scripts. 

---
# ClusterBased-WeightedClusters-Selector (CBWCS) Parameters

- **batch_size=[96, 384, 1536]**: the number of compounds to select each iteration. 

## Exploitation Parameters: 

- **exploitation_use_quantile_for_activity=[false, true]**: We prefer exploitation clusters with high activity scores. 
The threshold for highly active molecules is determined by **exploitation_activity_threshold**. However, the range of
activity scores depends on the model. RandomForest with small datasets typically have scores in the range [0,0.3], whereas
neural networks in the range [0,1]. So to remedy the model choice, if exploitation quantile is set to true, then the 
activity treshold will be based on the **exploitation_activity_threshold**'th quantile score (e.g. if **exploitation_activity_threshold=0.5** 
then threshold is the median score). If set to false, then will use raw **exploitation_activity_threshold** value.

- **exploitation_sample_actives_from_clusters=[false, true]**: when sampling compounds from exploitation clusers, all compounds (highly active and lowly active)
can be sampled if this is set to false. If set to true, then exploitation cluster sampling is limited to only highly active compounds as determined by **exploitation_activity_threshold**.

- **exploitation_activity_threshold=[0.0, 0.25, 0.5, 0.75, 1.0]**: compounds with activity scores >= this value are considered 'highly active'. 
This parameter works in conjunction with **exploitation_use_quantile_for_activity**, see description there.

- **exploitation_use_quantile_for_weight=[false, true]**: whether to use quantiling for **exploitation_weight_threshold**.

- **exploitation_weight_threshold=[0.0, 0.25, 0.5, 0.75, 1.0]**: clusters with exploitation weight >= this value are considered 'exploitation clusters'.
This parameter works in conjunction with **exploitation_use_quantile_for_weight**. 

- **exploitation_alpha=[0.0, 0.25, 0.5, 0.75, 1.0]**: This parameter is used in the exploitaiton weight calculation. Higher values gives more
weight to cluster activity and less weight to cluster labeled density (less values have the opposite effect). 

- **exploitation_dissimilarity_lambda=[0.0, 0.25, 0.5, 0.75, 1.0]**: the first exploitation cluster to be sampled is the highest weighted one. 
Iteratively, remaining exploitation clusters will then be re-weighted to account for their dissimilarity toward the already sampled clusters. We prefer 
clusters that have high exploitation weights but also dissimilar to clusters already sampled. Higher values of lambda give more weight to 
the dissimilarity and less weight to exploitation weight (e.g. lambda=0 completely ignores cluster dissimilarity). See issue #1 for more discussion.

- **use_intra_cluster_threshold_for_exploitation=[false, true]**: when sampling from a cluster, we sample compounds that are dissimilar. If this is set to true and 
**select_dissimilar_instances_within_cluster** is set to true, then we only consider compounds that are **intra_cluster_dissimilarity_threshold** apart. In other words, 
we want to sample compounds from a single cluster that are not close to each other (to cover as much as the cluster as possible). 

If this value is set to false and **select_dissimilar_instances_within_cluster** is set to true, then we sample compounds that are dissimilar but without considering 
**intra_cluster_dissimilarity_threshold**. For example, if all compounds within a cluster are within **intra_cluster_dissimilarity_threshold** distance of each other, 
we will still sample them dissimilarly. 


- **use_proportional_cluster_budget_for_exploitation=[false, true]**: This controls how much of the exploitation budget to allocate to each selected exploitation cluster. 
If set to true, then budgetting is proportionally split according to the number of compounds within the clusters (clusters with more compounds are given more budget). If set to false,
then budget is split equally. However, note that these cluster budgets are estimates, and during execution depending on **use_intra_cluster_threshold_for_exploitation**, the 
cluster budget of the currently sampled cluster may not be exhausted, and so the remaining budget will transfer to the next selected cluster. 

## Exploration Parameters:

- **exploration_strategy=["weighted", "random", "dissimilar"]**: _weighted_ uses the exploration weight method for selecting exploration clusters. _random_ will randomly (uniform) select 
leftover clusters (clusters that are not exploited) for exploration. _dissimilar_ randomly select the first exploration cluster, then iteratively select remaining exploration clusters that 
are dissimilar to what was already selected. Note that most of the exploration parameters work with _weighted_ method (ignored for others).


- **exploration_use_quantile_for_weight=[false, true]**: same as **exploitation_use_quantile_for_weight** but for exploration.

- **exploration_weight_threshold=[0.0, 0.25, 0.5, 0.75, 1.0]**: same as **exploitation_weight_threshold** but for exploration.

- **exploration_beta=[0.0, 0.25, 0.5, 0.75, 1.0]**: same as **exploitation_alpha** but for exploration.

- **exploration_dissimilarity_lambda=[0.0, 0.25, 0.5, 0.75, 1.0]**: same as **exploitation_dissimilarity_lambda** but for exploration.

- **use_intra_cluster_threshold_for_exploration=[false, true]**: same as **use_intra_cluster_threshold_for_exploitation** but for exploration.

- **use_proportional_cluster_budget_for_exploration=[false, true]**: same as **use_proportional_cluster_budget_for_exploitation** but for exploration.


## General Parameters

- **select_dissimilar_instances_within_cluster=[false, true]**: if set to true, samples compounds from clusters to ensure they are dissimilar. If set to false, then samples compounds 
from cluster randomly (uniformly). Note this method works in conjunction with **intra_cluster_dissimilarity_threshold**, **use_intra_cluster_threshold_for_exploitation**, 
and **use_intra_cluster_threshold_for_exploration**.

- **intra_cluster_dissimilarity_threshold=[0.0, 0.05, 0.1, 0.15, 0.2]**: used when sampling compounds from a cluster and  **select_dissimilar_instances_within_cluster** is set to true. 
We want to sample compounds from a single cluster that are not close to each other (to cover as much as the cluster as possible). 

- **feature_dist_func=["tanimoto_dissimilarity"]**: the metric used for calculating dissimilarity.

- **use_consensus_distance=[false, true]**: if set to true, uses a consensus featurization to compute a cluster feature, which is then used in calculating the cluster dissimilarity. 
For example, if using fingerprints, computes the consensus fingerprint for each cluster. If set to false, cluster dissimilarity is computed as the average dissimilarity of all compounds within 
the cluster.

- **uncertainty_method=[["least_confidence"], 
                        ["query_by_committee"], 
                        ["density_weight", 1.0, false],
                        ["density_weight", 1.0, true]]**: specifies the model's uncertainty calculation method.
                        
# ClusterBased-WeightedClusters-Selector (CBWCS) Parameters Distribution

Parameter space is in the 100s of billions. We plan on running parameter sweeps to find good parameters. Here we specify a distribution (probabilities) over the values of the parameters. 
This allows us to place preference over certain values based on our intuition. We can then run, say 100k, parameter jobs sampled from this preferred distribution.

```
  "nbs_params_probas": {
  
    "exploitation_use_quantile_for_activity": [0.5, 0.5], 
    "exploitation_sample_actives_from_clusters": [0.1, 0.9], # prefer to only select highly active compounds in exploitive clusters
    "exploitation_activity_threshold": [0.1, 0.1, 0.3, 0.5, 0.0], # prefer highly active clusters. 0.0 disallows the value=1.0 resultin in NO exploitive clusters.
    "exploitation_use_quantile_for_weight": [0.5, 0.5],
    "exploitation_weight_threshold": [0.1, 0.1, 0.35, 0.35, 0.1], # prefer highly exploitive weights
    "exploitation_alpha": [0.2, 0.2, 0.2, 0.2, 0.2], # not sure of effect
    "exploitation_dissimilarity_lambda": [0.2, 0.2, 0.2, 0.2, 0.2], # not sure of effect
    "use_intra_cluster_threshold_for_exploitation": [0.2, 0.8], # for exploitive clusters, prefer far apart dissimilar compounds.
    "use_proportional_cluster_budget_for_exploitation": [0.8, 0.2], # for exploitive clusters, prefer equal budgetting.
    
    "exploration_strategy": [0.4, 0.3, 0.3],
    "exploration_use_quantile_for_weight": [0.5, 0.5],
    "exploration_weight_threshold": [0.1, 0.1, 0.35, 0.35, 0.1],
    "exploration_beta": [0.2, 0.2, 0.2, 0.2, 0.2],
    "exploration_dissimilarity_lambda": [0.2, 0.2, 0.2, 0.2, 0.2],
    "use_intra_cluster_threshold_for_exploration": [0.8, 0.2], # for explorative clusters, don't take compounds distances into account
    "use_proportional_cluster_budget_for_exploration": [0.2, 0.8], # for explorative clusters, prefer proportional budgeting. Want to cover more of low coverage clusters.
    
    "select_dissimilar_instances_within_cluster": [0.25, 0.75], # prefer to incorporate dissimilarity when selecting compounds in cluster to cover more cluster space.
    "intra_cluster_dissimilarity_threshold": [0.0, 0.3, 0.25, 0.25, 0.2],
    "feature_dist_func": [1.0],
    "use_consensus_distance": [0.5, 0.5],
    
    "uncertainty_method": [0.25, 0.25, 0.25, 0.25] # not sure of best method
  }
}
```