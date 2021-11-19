import numpy as np
from util.arg_utils import get_args, extract_args_from_json

args, device = get_args()  # get arguments from command line
np.random.seed(seed=20190508)  # set the seeds for the experiment

import torch
import sys
from torch.utils.data import Subset

from models.factor_analysis import FactorAnalysis
from models.variational_distribution import GaussianVarDistr, SharedGaussianVarDistr, SharedGaussianVarDistrRelaxed
from data.frey_faces import IncompleteFreyFacesDataset
from data.toy_set import ToyDataset
from common.imputation import impute_with_variational_means
from util.mutable_subset import MutableSubset
import copy
import os
import json

torch.manual_seed(seed=20190508)  # set the seeds for the experiment

def load_dataset(missing_fraction, args, seed=20190508):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load dataset
    # Prepare dataset
    if args.dataset == 'ff_gen_processed':
        print('Using preprocessed generated Frey Faces.')
        dataset = IncompleteFreyFacesDataset(root=args.dataset_root,
                                            missing_fraction=missing_fraction,
                                            preprocess=True,
                                            generated=True)
    elif args.dataset == 'frey_faces_processed':
        print('Using preprocessed Frey Faces.')
        dataset = IncompleteFreyFacesDataset(root=args.dataset_root,
                                            missing_fraction=missing_fraction,
                                            preprocess=True,
                                            generated=False)
    elif args.dataset == 'frey_faces_not_processed':
        print('Using *not* preprocessed Frey Faces.')
        dataset = IncompleteFreyFacesDataset(root=args.dataset_root,
                                            missing_fraction=missing_fraction,
                                            preprocess=False,
                                            generated=False)
    # elif args.dataset == 'toy_set':
    #     dataset = ToyDataset(root=args.dataset_root,
    #                         missingness=missing_fraction,
    #                         missingness_type=args.missingness_type,
    #                         version=1)
    # elif args.dataset == 'toy_set2':
    #     dataset = ToyDataset(root=args.dataset_root,
    #                         missingness=missing_fraction,
    #                         missingness_type=args.missingness_type,
    #                         version=2)
    # elif args.dataset == 'toy_set3':
    #     dataset = ToyDataset(root=args.dataset_root,
    #                         missingness=missing_fraction,
    #                         missingness_type=args.missingness_type,
    #                         version=3)
    else:
        print('No such dataset available!')
        sys.exit()

    # Train/Val split
    train_fraction = 0.8
    dataset_size = len(dataset)
    indices = np.random.permutation(range(dataset_size))
    split = int(np.floor(train_fraction * dataset_size))
    train_indices, val_indices = indices[:split], indices[split:]

    # Create validation set
    train_dataset = MutableSubset(dataset, train_indices)
    val_dataset = MutableSubset(dataset, val_indices)

    return train_dataset, val_dataset


def impute_dataset_with_variational_means(dataset,
                                        missingness,
                                        exp_subdir,
                                        var_model,
                                        var_epoch,
                                        imputations=1,
                                        switch_imputations=999999999999999999999999999999999):

    dataset = copy.deepcopy(dataset)

    # Load variational model config
    variational_args = extract_args_from_json('./experiment_configs/{}/learning_experiments/{:d}/{}.json'.format(exp_subdir, int(missingness), var_model))

    # Impute with variational mean
    impute_with_variational_means(dataset, variational_args, epoch=var_epoch,
                                  device=device, root_dir='./', exp_group=variational_args.exp_group,
                                  repeat=imputations, switch_imputations=switch_imputations)

    return dataset

def create_imputed_datasets(dataset, missingness, exp_subdir, method, epoch, imputations=1, switch_imputations=999999999999999999999999999999999):
    dataset = impute_dataset_with_variational_means(dataset,
                                                    missingness,
                                                    exp_subdir,
                                                    var_model=method,
                                                    var_epoch=epoch,
                                                    imputations=imputations,
                                                    switch_imputations=switch_imputations)

    return dataset

def comp_imputation_mean_sq_err(true_dataset, imputed_dataset):
    X, M, _ = true_dataset[:]
    X_imp, M_imp, _ = imputed_dataset[:]

    err = (X - X_imp)**2
    err_sum = (err * (1-M_imp)).sum(axis=0)
    missing_per_dimension = np.count_nonzero((1-M_imp), axis=0)
    return err_sum/missing_per_dimension

def compute_imputation_mse_dict(args):
    imputation_mse_dict = {'mse': [], 'std_err': []}
    with torch.no_grad():
        missingness = [0.1667, 0.3333, 0.5, 0.6667, 0.8333]
        for i, m in enumerate([1, 2, 3, 4, 5]):
            _, val_dataset = load_dataset(missingness[m-1], args, seed=20190508)

            temp_mses = []
            for k in range(args.K):
                # Keep the original dataset and missingness mask for the first comparison
                if k > 0:
                    # Generate another validation dataset
                    _, _, I = val_dataset[:]
                    val_dataset[I] = val_dataset.dataset.generate(len(val_dataset))
                    # Shuffle the missingness mask for other comparisons
                    val_dataset.dataset.init_missingness_mask()

                imputed_dataset = create_imputed_datasets(val_dataset,
                                        m,
                                        exp_subdir=args.exp_group,
                                        method=args.method,
                                        epoch=args.epochs[i],
                                        imputations=args.imputations,
                                        switch_imputations=args.switch_imputations)

                temp_mses.append(np.mean(comp_imputation_mean_sq_err(val_dataset, imputed_dataset)))

            imputation_mse_dict['mse'].append(np.mean(temp_mses))
            imputation_mse_dict['std_err'].append(np.std(temp_mses, ddof=1)*(args.K**-0.5))

    return imputation_mse_dict

imputation_mse_dict = compute_imputation_mse_dict(args)
print(imputation_mse_dict)

output_dir = './imputation_stats/{}/'.format(args.exp_group)
# Create necessary directories
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

summary_filename = os.path.join(output_dir, args.experiment_name)
with open(summary_filename, 'w') as f:
    f.write(json.dumps(imputation_mse_dict))
