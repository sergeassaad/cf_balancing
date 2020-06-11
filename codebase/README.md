# Counterfactual Regression with Balancing Weights

This repository is the codebase for Counterfactual Regression with Balancing Weights.
This code is implemented in Python using TensorFlow 1.13.1.

The code is inspired by https://github.com/clinicalml/cfrnet

# Code

The codebase's main functions follow the functions of CFRNet.

Namely:

- cfr_param_search.py runs a random hyperparameter search given a text file with lists of hyperparameters. All the config files used for this work can be found in the folder configs/neurips/
Usage:
```
python cfr_param_search.py <config_file> <num_runs>
```

- evaluate.py evaluates the results of cfr_param_search.py
Usage:
```
python evaluate.py <config_file> [overwrite]
```

- ablation.py is akin to cfr_param_search.py, but it allows to loop over an ablation variable (in our case the different weight schemes of interest) so as to have a fair comparison for the ablation variable.
Usage:
```
python ablation.py <config_file> <num_runs> <ablation_var>
```
where <ablation_var> is a string (in our case we use "weight_scheme") indicating the ablation variable.

## IHDP

For hyperparameter tuning, we used the file configs/neurips/ihdp100.txt, as:

``` 
python ablation.py configs/neurips/ihdp100.txt 100
```

Then, we used the notebook process_ihdp_results.ipynb to pick the best model based on 1-nearest-neighbor imputation on the validation set.

Finally, we run the evaluation script for IHDP1000 on the best configuration for each weight scheme, as:
```
python evaluate.py configs/neurips/ihdp1000/<weight_scheme>.txt 1
```
where <weight_scheme> can be one of IPW, OW, MW, or TruncIPW.

## ACIC 2016

For the ACIC2016 dataset, we tuned the model based on the first 10/77 datasets (with 1 repetition each). The hyperparameter ranges can be found in configs/neurips/acic2016.txt

Similar to the IHDP dataset, we then pick the best model for each weight scheme, and evaluate on 77 datasets x 10 repetitions.

## Toy experiment

To generate the toy data used in the paper, please use Toy_data_SNR_2.ipynb
The hyperparameter file can be found at configs/neurips/toy.txt

We call:
```
python cfr_param_search.py configs/neurips/toy.txt 10000
```
Note: for the toy experiment there are 11 values of gamma x 3 confounding scenarios x 6 alpha values x 4 weight schemes = 792 total configurations

We then process the toy experiment results using process_toy_results_SNR.ipynb
