# Variational Gibbs inference (VGI)

This repository contains the research code for

> Simkus, V., Rhodes, B., Gutmann, M. U., 2021. Variational Gibbs inference for statistical model estimation from incomplete data.

The code is shared for reproducibility purposes and is not intended for production use. It should also serve as a reference implementation for anyone wanting to use VGI for model estimation from incomplete data.

## Abstract

Statistical models are central to machine learning with broad applicability across a range of downstream tasks. The models are typically controlled by free parameters that are estimated from data by maximum-likelihood estimation. However, when faced with real-world datasets many of the models run into a critical issue: they are formulated in terms of fully-observed data, whereas in practice the datasets are plagued with missing data. The theory of statistical model estimation from incomplete data is conceptually similar to the estimation of latent-variable models, where powerful tools such as variational inference (VI) exist. However, in contrast to standard latent-variable models, parameter estimation with incomplete data often requires estimating exponentially-many conditional distributions of the missing variables, hence making standard VI methods intractable. We address this gap by introducing variational Gibbs inference (VGI), a new general-purpose method to estimate the parameters of statistical models from incomplete data.

## VGI demo

We invite the readers of the paper to also see the Jupyter [notebook](https://nbviewer.org/github/vsimkus/variational-gibbs-inference/blob/main/notebooks/VGI_demo.ipynb), where we demonstrate VGI on two statistical models and animate the learning process to help better understand the method.

Below is an animation from the notebook of a Gaussian Mixture Model fitted from incomplete data using the VGI algorithm (left), and the variational Gibbs conditional approximations (right) throughout iterations.

<https://user-images.githubusercontent.com/5730052/142662708-5554b1e6-1d62-4848-a5bd-ab0000e6adcd.mp4>

## Dependencies

Install python dependencies from conda and the `cdi` project package with

```bash
conda env create -f environment.yml
conda activate cdi
python setup.py develop
```

If the dependencies in `environment.yml` change, update dependencies with

```bash
conda env update --file environment.yml
```

## Summary of the repository structure

### Data

All data used in the paper are stored in [`data`](./data/) directory and the corresponding data loaders can be found in [`cdi/data`](./cdi/data/) directory.

### Method code

The main code to the various methods used in the paper can be found in [`cdi/trainers`](./cdi/trainers/) directory.

* [`trainer_base.py`](./cdi/trainers/trainer_base.py) implements the main data loading and preprocessing code.
* [`variational_cdi.py`](./cdi/trainers/variational_cdi.py) and [`cdi.py`](./cdi/trainers/cdi.py) implement the key code for variational Gibbs inference (VGI).
* [`mcimp.py`](./cdi/trainers/mcimp.py) implements the code for variational block-Gibbs inference (VBGI) used in the VAE experiments.
* The other scripts in [`cdi/trainers`](./cdi/trainers/) implement the comparison methods and variational conditional pre-training.

### Statistical models

The code for the statistical (factor analysis, VAEs, and flows) and the variational models are located in [`cdi/models`](./cdi/models/).

### Configuration files

The [`experiment_configs`](./experiment_configs/) directory contains the configuration files for all experiments. The config files include all the hyperparameter settings necessary to reproduce our results. The config files are in a json format. They are passed to the main running script as a command-line argument and values in them can be overriden with additional command-line arguments.

### Run scripts

[`train.py`](./train.py) is the main code we use to run the experiments, and [`test.py`](./test.py) is the main script to produce analysis results presented in the paper.

### Analysis code

The Jupyter notebooks in [`notebooks`](./notebooks/) directory contain the code which was used to analysis the method and produce figures in the paper. You should also be able to use these notebooks to find the corresponding names of the config files for the experiments in the paper.

## Running the code

Before running any code you'll need to activate the `cdi` conda environment (and make sure you've installed the dependencies)

```bash
conda activate cdi
```

### Model fitting

To train a model use the `train.py` script, for example, to fit a rational-quadratic spline flow on 50% missing MiniBooNE dataset

```bash
python train.py --config=experiment_configs/flows_uci/learning_experiments/3/rqcspline_miniboone_chrqsvar_cdi_uncondgauss.json
```

Any parameters set in the config file can be overriden by passing additionals command-line arguments, e.g.

```bash
python train.py --config=experiment_configs/flows_uci/learning_experiments/3/rqcspline_miniboone_chrqsvar_cdi_uncondgauss.json --data.total_miss=0.33
```

#### Optional variational model warm-up

Some VGI experiments use variational model "warm-up", which pre-trains the variational model on observed data as probabilistic regressors. The experiment configurations for these runs will have `var_pretrained_model` set to the name of the pre-trained model. To run the corresponding pre-training script run, e.g.

```bash
python train.py --config=experiment_configs/flows_uci/learning_experiments/3/miniboone_chrqsvar_pretraining_uncondgauss.json
```

## Running model evaluation

For model evaluation use [`test.py`](./test.py) with the corresponding test config, e.g.

```bash
python test.py --test_config=experiment_configs/flows_uci/eval_loglik/3/rqcspline_miniboone_chrqsvar_cdi_uncondgauss.json
```

This will store all results in a file that we then analyse in the provided notebook.

For the VAE evaluation, where variational distribution fine-tuning is required for test log-likelihood evaluation use [`retrain_all_ckpts_on_test_and_run_test.py`](./retrain_all_ckpts_on_test_and_run_test.py).
