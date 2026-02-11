# multi-objective-influence-maximization
This repository contains the code for the publication "[Detecting bias in algorithms used to disseminate information in social networks and mitigating it using multi-objective optimization](https://academic.oup.com/pnasnexus/article/4/10/pgaf291/8292699?login=true)" by V. Sekara, I. Dotu, M. Cebrian, E. Moro, and M. Garcia-Herranz.

## Contents

This repository contains multiple files, below we provide a short guide to files and explain their usage.
* __core_functions.py__: Contains the core functions for running Independent Cascade Models (ICM), loading networks from files, loading infection/transmission probabilities from files, and the various heuristics to identify influencer nodes. 
* __create_synthetic_networks.ipynb__: Code to generate synthetic Scale-free and Erdos–Renyi networks. 
* __infer_pc.ipynb__: Code to determine the critical transmission probability for synthetic networks
* __infer_pc_empirical.ipynb__: Code to determine the critical transmission probability for empirical networks
* __information_spread_empirical_ensemble.ipynb__: Notebook to simulate information cascades in empirical networks given seed sets identified by the traditional influence heuristics. 
* __lastMileGAsSetsFast.py__: Multi-objective code to identify fairer seed sets.
* __model_topoligical_features_empirical.ipynb__: Notebook that builds Machine Learning models to eestimate which topological features detrmine whether a node will be vulunerable.
* __run_icm_for_fair_solutions.ipynb__: Code to run ICMs for the seed sets identified by _lastMileGAsSetsFast.py_ 
