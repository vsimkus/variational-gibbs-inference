import os
import sys
import time
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from jsonargparse import ArgumentParser

from cdi.models.regression import UnivariateRegression
from cdi.util.data.data_augmentation_dataset import DataAugmentation, \
                                                    collate_augmented_samples
from cdi.util.regression_sampler import RegressionSampler
from cdi.util.stats_utils import save_statistics
from cdi.util.arg_utils import parse_bool


class RegressionTrainer(nn.Module):
    """
    Trains univariate regression models for each variable given all others.
    Starting from 0-th dimension, trains the regression model, then imputes
    the missing values at 0-th dimension, and moves to the next dimension.
    """
    def __init__(self, num_models, model_arch, experiment_name, num_epochs,
                 batch_size, train_dataset, val_dataset,
                 weight_decay_coefficient, learning_rate, device,
                 early_stop_epoch_thresh, continue_training=False,
                 impute=True, input_missing_vectors=False, root_dir='.',
                 exp_group=''):
        """
        Args:
            num_models (int): Number of regression models to train
                (usually one for each dimension)
            model_arch (dict): Regression model configuration
            experiment_name (string): Name of the experiment used for storing
                model parameters
            num_epochs (int): number of epochs to train
            batch_size (int): batch size to be used in training
            train_dataset (iterable): iterable dataset that return x - inputs,
                and m - missingness mask
            val_dataset (iterable): iterable dataset that return x - inputs,
                and m - missingness mask
            weight_decay_coefficient (float): weight decay used in Adam
            learning_rate (float): learning rate using in Adam optimiser
            device (torch.device): device to perform the computations on
            early_stop_epoch_thresh (int): number of epochs to continue if no
                improvement in validation loss before early stopping (for each
                regression function)
            continue_training (bool, default=False): Whether to continue
                training from where it was left-off, or start anew.
                If True, loads all trained regression models at their best
                validation loss, and imputes values for all values that are
                already trained.
            impute (bool, default=True): Whether it should impute the dataset
                as it trains, or not.
            input_missing_vectors (bool, default=False): Whether the binary
                missingness masks should be input into the regression
                functions.
            root_dir (string or path): main directory of the project
            exp_group (string): experiment group name, used to separate out
                models in trained_models directory.
        """
        super(RegressionTrainer, self).__init__()

        self.experiment_name = experiment_name
        self.device = device

        # Set data
        self.impute = impute
        self.input_missing_vectors = input_missing_vectors
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stop_epoch_thresh = early_stop_epoch_thresh

        self.num_models = num_models
        self.model_arch = model_arch
        self.learning_rate = learning_rate
        self.weight_decay_coefficient = weight_decay_coefficient

        # Generate the directory names
        self.experiment_root = os.path.join(os.path.abspath(root_dir),
                                            'trained_models',
                                            exp_group)
        self.experiment_folder = os.path.abspath(
                                            os.path.join(self.experiment_root,
                                                         experiment_name))
        self.experiment_logs = os.path.abspath(
                                        os.path.join(self.experiment_folder,
                                                     'logs'))
        self.experiment_saved_models = os.path.abspath(
                                        os.path.join(self.experiment_folder,
                                                     'saved_models'))
        print(self.experiment_folder, self.experiment_logs)
        # Create necessary directories
        if not os.path.exists(self.experiment_root):
            os.makedirs(self.experiment_root)

        if not os.path.exists(self.experiment_folder):
            os.makedirs(self.experiment_folder)

        if not os.path.exists(self.experiment_logs):
            os.makedirs(self.experiment_logs)

        if not os.path.exists(self.experiment_saved_models):
            os.makedirs(self.experiment_saved_models)

        # Set best models to be at -1 and loss at inf since we are just starting
        self.best_val_model_idx = defaultdict(def_model_idx)
        self.best_val_model_loss = defaultdict(def_model_loss)

        # Continue from where left-off
        self.continue_training = continue_training
        if self.continue_training:
            try:
                # Load state
                self.state = load_state(model_save_dir=self.experiment_saved_models,
                                        state_save_name='train_state')
                self.starting_dimension = self.state['current_dim']
                self.starting_epoch = self.state['current_epoch_idx'] + 1
                self.best_val_model_idx = self.state['best_val_model_idx']
                self.best_val_model_loss = self.state['best_val_model_loss']
                # self.starting_epoch = self.best_val_model_idx[self.starting_dimension]

                # Impute the dimensions for which models are fully trained.
                if self.impute:
                    for dim in range(0, self.num_models):
                        # Load only trained models and not the current one
                        if self.best_val_model_idx[dim] != -1 and dim != self.starting_dimension:
                            print('Loading {}-th model.'.format(dim))
                            unwrapped_model = self.create_model()

                            # Load model parameters
                            load_model(model_save_dir=os.path.join(self.experiment_saved_models, str(dim)),
                                       model_save_name='train_model',
                                       model_idx=self.best_val_model_idx[dim],
                                       model=unwrapped_model)

                            # Send model to device
                            model = send_model_to_device(unwrapped_model, self.device)

                            print('Imputing {}-th dimension.'.format(dim))
                            impute_dimension_with_regression_predictions(self.train_dataset,
                                                                         model,
                                                                         dim,
                                                                         self.batch_size,
                                                                         self.device,
                                                                         self.input_missing_vectors)
                            impute_dimension_with_regression_predictions(self.val_dataset,
                                                                         model,
                                                                         dim,
                                                                         self.batch_size,
                                                                         self.device,
                                                                         self.input_missing_vectors)
            except Exception as e:
                traceback.print_exc()
                print('Model cannot be found in {}!'.format(self.experiment_saved_models))
                sys.exit()
        else:
            self.starting_dimension = 0
            self.starting_epoch = 0
            self.state = dict()

    @staticmethod
    def add_regression_trainer_args(parent_parser):
        # Add regression baseline parameters
        parser = ArgumentParser(parser_mode='jsonnet',
                                parents=[parent_parser],
                                add_help=False)
        parser.add_argument('--num_regression_fns',
                            type=int, required=True,
                            help=('Number of regression models to train '
                                  '(usually one for each dimension)'))
        parser.add_argument('--learning_rate',
                            type=float, required=True,
                            help=('The learning rate using in Adam '
                                  'optimiser for the regressors.'))
        parser.add_argument('--weight_decay_coefficient',
                            type=float, required=True,
                            help=('The weight decay used in Adam '
                                  'optimiser for the regressors.'))
        parser.add_argument('--input_missing_vectors',
                            type=parse_bool, required=True,
                            help=('Whether the binary missingness '
                                  'masks should be input into the '
                                  'regression functions.'))
        parser.add_argument('--early_stop_epoch_thresh',
                            type=int, required=True,
                            help=('Number of epochs to continue if no '
                                  'improvement in validation loss before '
                                  'early stopping (for each regression '
                                  'function)'))
        parser.add_argument('--continue_training',
                            type=parse_bool, default=False,
                            help=('Whether to continue training from where '
                                  'it was left-off, or start anew. If True, '
                                  'loads all trained regression models at '
                                  'their best validation loss, and imputes '
                                  'values for all values that are already '
                                  'trained.'))
        parser = UnivariateRegression.add_model_args(parser)

        return parser

    def save_state(self, model_save_dir, state_save_name, state):
        """
        Save training state.
        """
        torch.save(state, f=os.path.join(model_save_dir, state_save_name))

    def save_model(self, model_save_dir, model_save_name, model_idx, model):
        """
        Saves the specified regression model.
        """
        # Create model directory if it doesn't exist
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

        model_state = model.state_dict()
        torch.save(model_state, f=os.path.join(model_save_dir, '{}_{}'.format(model_save_name, str(model_idx))))

    def create_model(self):
        unwrapped_model = UnivariateRegression(self.model_arch)
        unwrapped_model.reset_parameters()
        return unwrapped_model

    def run_experiment(self):
        """
        Performs iterative training of the regression models using the `run_train_iter` method.
        First starts training 0th dimension, given the others, then imputes the values at the 0th dimension,
        and then trains the next regression model.
        """
        for i, d in enumerate(range(self.starting_dimension, self.num_models)):
            print('Training {}-th regression function (out of {}).'.format(d, self.num_models))
            current_dimension_metrics = defaultdict(list)
            dim_start_time = time.time()

            # Create dataloader with custom sampler that will filter out samples that are missing a value at the d-th dimension.
            train_data = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                collate_fn=collate_augmented_samples,
                num_workers=0,
                sampler=RegressionSampler(self.train_dataset,
                                          dim=d,
                                          shuffle=True,
                                          omit_missing=True))
            val_data = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                # collate_fn=collate_augmented_samples,
                num_workers=0,
                sampler=RegressionSampler(self.val_dataset,
                                          dim=d,
                                          shuffle=True,
                                          omit_missing=True))

            # Create model for the current dimension
            unwrapped_model = self.create_model()
            if self.continue_training and i == 0:  # If we're resuming, load the last model
                load_model(model_save_dir=os.path.join(self.experiment_saved_models, str(d)),
                           model_save_name='train_model',
                           model_idx='latest',
                           model=unwrapped_model)
            model = send_model_to_device(unwrapped_model, self.device)

            # Prepare Adam optimiser for the model
            optimiser = optim.Adam(model.parameters(),
                                   amsgrad=False,
                                   lr=self.learning_rate,
                                   weight_decay=self.weight_decay_coefficient)
            # optimiser = optim.LBFGS(params=model.parameters(),
            #                         lr=self.learning_rate)

            # If we're just starting, then start at start_epoch, otherwise at 0
            start_epoch = self.starting_epoch if i == 0 else 0
            for j, epoch_idx in enumerate(range(start_epoch, self.num_epochs)):
                epoch_start_time = time.time()
                current_epoch_metrics = defaultdict(list)

                with tqdm.tqdm(total=len(train_data)) as pbar_train:  # create a progress bar for training
                    for X, M, I in train_data:
                        # Targets are the d-th dimension
                        Y_d = X[:, d]

                        # Remove d-th feature from X
                        feature_indices = np.arange(X.shape[-1])
                        feature_indices = feature_indices[feature_indices != d]
                        X_d = X[:, feature_indices]

                        M_d = None
                        if self.input_missing_vectors:
                            M_d = M[:, feature_indices]

                        # take a training iter step
                        metrics = self.run_train_iter(X=X_d, Y=Y_d, model=model, optimiser=optimiser, M=M_d)

                        # Append metrics from the current iteration
                        for key, value in metrics.items():
                            current_epoch_metrics['train_{}'.format(key)].append(value)

                        # Format progress bar description
                        description = (', ').join(['{}: {:.4f}'.format(key, value) for key, value in metrics.items()])
                        pbar_train.set_description(description)
                        pbar_train.update(1)

                train_epoch_finish = time.time()

                with tqdm.tqdm(total=len(val_data)) as pbar_val:  # create a progress bar for validation
                    for X, M, I in val_data:
                        with torch.no_grad():
                            # Targets are the d-th dimension
                            Y_d = X[:, d]

                            # Remove d-th feature from X
                            feature_indices = np.arange(X.shape[-1])
                            feature_indices = feature_indices[feature_indices != d]
                            X_d = X[:, feature_indices]

                            M_d = None
                            if self.input_missing_vectors:
                                M_d = M[:, feature_indices]

                            # run a validation iter
                            metrics = self.run_evaluation_iter(X=X_d, Y=Y_d, model=model, M=M_d)

                            # Append metrics from the current iteration
                            for key, value in metrics.items():
                                current_epoch_metrics['val_{}'.format(key)].append(value)

                            # Format progress bar description
                            description = (', ').join(['{}: {:.4f}'.format(key, value) for key, value in metrics.items()])
                            pbar_val.set_description(description)
                            pbar_val.update(1)

                val_loss = np.mean(current_epoch_metrics['val_loss'])
                if val_loss < self.best_val_model_loss[d]:  # if current epoch's val loss is greater than the saved best val loss then
                    self.best_val_model_loss[d] = val_loss  # set the best val model loss to be current epoch's val loss
                    self.best_val_model_idx[d] = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx

                # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.
                for key, value in current_epoch_metrics.items():
                    current_dimension_metrics[key].append(np.mean(value))

                # save statistics to stats file.
                current_dimension_metrics['curr_epoch'].append(epoch_idx)
                current_dimension_metrics['train_time'].append(train_epoch_finish - epoch_start_time)
                current_dimension_metrics['val_time'].append(time.time() - train_epoch_finish)
                save_statistics(experiment_log_dir=self.experiment_logs,
                                filename='summary{}.csv'.format(d),
                                stats_dict=current_dimension_metrics,
                                current_epoch=j,
                                continue_from_mode=True if (start_epoch != 0 or j > 0) else False)

                # create a string to report our epoch metrics
                out_string = "_".join(
                    ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_metrics.items()])
                epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
                epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
                print("\nEpoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")

                if torch.cuda.is_available():
                    print('CUDA max allocated memory: {}, max cached memory: {}.'.format(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_cached()))

                # Save state of regression function training
                self.state['current_dim'] = d
                self.state['current_epoch_idx'] = epoch_idx
                self.state['best_val_model_loss'] = self.best_val_model_loss
                self.state['best_val_model_idx'] = self.best_val_model_idx
                self.save_state(model_save_dir=self.experiment_saved_models,
                                state_save_name='train_state',
                                state=self.state)

                # Save state of current dimension function
                self.save_model(model_save_dir=os.path.join(self.experiment_saved_models, str(d)),
                                model_save_name='train_model',
                                model_idx=epoch_idx,
                                model=unwrapped_model)
                self.save_model(model_save_dir=os.path.join(self.experiment_saved_models, str(d)),
                                model_save_name='train_model',
                                model_idx='latest',
                                model=unwrapped_model)

                if self.best_val_model_idx[d] + self.early_stop_epoch_thresh < epoch_idx:
                    print('Breaking early for {}-th regression fn.'.format(d))
                    break

            # Load best model for the current dimension
            load_model(model_save_dir=os.path.join(self.experiment_saved_models, str(d)),
                       model_save_name='train_model',
                       model_idx=self.best_val_model_idx[d],
                       model=model)

            # Remove model parameters that were not the best.
            for f in os.listdir(os.path.join(self.experiment_saved_models, str(d))):
                if f not in ['train_model_{}'.format(self.best_val_model_idx[d]), 'train_model_latest']:
                    os.remove(os.path.join(self.experiment_saved_models, str(d), f))

            # Impute missing values at d-th dimension.
            if self.impute:
                print('Imputing {}-th dimension.'.format(d))
                impute_dimension_with_regression_predictions(dataset=self.train_dataset,
                                                             model=model,
                                                             dim=d,
                                                             batch_size=self.batch_size,
                                                             device=self.device,
                                                             input_missing_vectors=self.input_missing_vectors)
                impute_dimension_with_regression_predictions(dataset=self.val_dataset,
                                                             model=model,
                                                             dim=d,
                                                             batch_size=self.batch_size,
                                                             device=self.device,
                                                             input_missing_vectors=self.input_missing_vectors)

            # calculate time taken for current dimension
            dim_time = time.time() - dim_start_time
            print('{}-th dimension completed in {:.4f} seconds'.format(d, dim_time))

        # Save final data
        X, M, _ = self.train_dataset[:]
        data_tensors = {
            'X': X,
            'M': M
        }

        # Store to file
        filepath = os.path.join(self.experiment_logs, 'tensors')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filepath = os.path.join(filepath, 'train_data_final.npz')
        print(f'Saving final data to {filepath}')
        np.savez_compressed(filepath, **data_tensors)

        return self.state

    def run_train_iter(self, X, Y, model, optimiser, M=None):
        self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)

        if type(X) is np.ndarray:
            X = torch.Tensor(X).float()
        if type(Y) is np.ndarray:
            Y = torch.Tensor(Y).float()
        if M is not None and type(M) is np.ndarray:
            M = torch.Tensor(M).float()

        X = X.to(device=self.device)
        Y = Y.to(device=self.device)
        if M is not None:
            M = M.to(device=self.device)

        def closure():
            # Compute loss
            out = model.forward(X, M).squeeze()
            loss = F.mse_loss(out, Y)

            # Update
            optimiser.zero_grad()  # set all weight grads from previous training iters to 0
            loss.backward()  # backpropagate to compute gradients for current iter loss
            return loss

        loss = optimiser.step(closure=closure)

        metrics = {
            'loss': loss.data.detach().cpu().numpy()
        }
        return metrics

    def run_evaluation_iter(self, X, Y, model, M=None):
        self.eval() # sets the system to validation mode

        metrics = evaluate_imputation_loss(X, Y, model, device=self.device, M=M)

        return metrics


# Workaround for torch.save not being able to pickle lambdas.
# Default values for best_val_model_idx, and best_val_model_loss
def def_model_idx():
    return -1


def def_model_loss():
    return float('inf')


def load_state(model_save_dir, state_save_name):
    """
    Load training state for continuing training.
    """
    file_path = os.path.join(model_save_dir, state_save_name)
    state = torch.load(f=file_path)

    return state


def load_model(model_save_dir, model_save_name, model_idx, model, disable_print=False):
    """
    Load regression model at the specified epoch.
    """
    if not disable_print:
        print('Loading regression model {}/{}, at epoch {}.'.format(model_save_dir, model_save_name, model_idx))

    file_path = os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx)))
    if torch.cuda.is_available():
        model_state = torch.load(f=file_path)
    else:
        model_state = torch.load(f=file_path, map_location='cpu')

    model.load_state_dict(model_state)


def load_models(unwrapped_models, experiment_name, root_dir='.', exp_group='', disable_print=False):
    """
    Load all regression models that are already trained according to the state,
    and load them from their *best* validation performance.
    """
    experiment_root = os.path.join(os.path.abspath(root_dir), 'trained_models', exp_group)
    experiment_folder = os.path.abspath(os.path.join(experiment_root, experiment_name))
    experiment_saved_models = os.path.abspath(os.path.join(experiment_folder, 'saved_models'))

    state = load_state(model_save_dir=experiment_saved_models,
                                            state_save_name='train_state')
    best_val_model_idx = state['best_val_model_idx']

    for d, model in enumerate(unwrapped_models):
        # Only load if a model for this dimension was already trained.
        if best_val_model_idx[d] != -1:
            load_model(model_save_dir=os.path.join(experiment_saved_models, str(d)),
                    model_save_name='train_model',
                    model_idx=best_val_model_idx[d],
                    model=model,
                    disable_print=disable_print)


def send_model_to_device(model, device):
    """
    Sends model to the device used for training, and in case of cuda, parallelises.
    Args:
        model (nn.Module): model
        device (torch.device): device to send to (CPU or GPU)
    Returns:
        model: If device is CPU then the same as input. Otherwise, if cuda and
            there are more than one devices, then nn.DataParallel
    """
    # Send to cuda device if available
    if torch.cuda.device_count() > 1:
        model.to(device)
        model = nn.DataParallel(module=model)
    else:
        model.to(device)

    return model


def impute_dimension_with_regression_predictions(dataset, model, dim, batch_size, device, input_missing_vectors):
    """
    Imputes missing values of the selected dimension of the given dataset using the regression model.
    Args:
        dataset (iterable): iterable dataset to be imputed, that returns x - inputs, and m - missingness mask
        models (nn.Module): regression model for imputation of the d-th dimension
        dim (int): dimension to be imputed
        batch_size (int): batch_size to be used during imputation.
        device (torch.device): device to perform the computations on
        input_missing_vectors (bool): Whether to input missing vectors to regression functions or not
    """
    if isinstance(dataset, DataAugmentation):
        collate_fn = collate_augmented_samples
    else:
        collate_fn = None

    data = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        sampler=RegressionSampler(dataset,
                                  dim=dim,
                                  shuffle=False,
                                  omit_missing=False))  # Iterate over missing-values at d-th dimension only
    if len(data) == 0:
        # Nothing to impute here.
        return

    with torch.no_grad():
        for batch in data:
            X, M, I = batch[:3]
            if type(X) is np.ndarray:
                X = torch.Tensor(X).float()
            X = X.to(device=device)

            # Remove d-th feature from X batch
            feature_indices = np.arange(X.shape[-1])
            feature_indices = feature_indices[feature_indices != dim]
            X_d = X[:, feature_indices]

            M_d = None
            if input_missing_vectors:
                M_d = M[:, feature_indices]

            # Impute at d-th dimension
            dataset[I, dim] = model.forward(X_d, M_d).squeeze().detach().cpu()


def evaluate_imputation_loss(X, Y, model, device, M=None, postprocess_fn=None):
    if type(X) is np.ndarray:
        X = torch.Tensor(X).float()
    if type(Y) is np.ndarray:
        Y = torch.Tensor(Y).float()
    if M is not None and type(M) is np.ndarray:
        M = torch.Tensor(M).float()

    X = X.to(device=device)
    Y = Y.to(device=device)
    if M is not None:
        M = M.to(device=device)

    # Compute loss
    out = model.forward(X, M).squeeze()
    if postprocess_fn is not None:
        Y = postprocess_fn(Y)
        out = postprocess_fn(out)

    loss = F.mse_loss(out, Y)

    metrics = {
        'loss': loss.data.detach().cpu().numpy()
    }
    return metrics
