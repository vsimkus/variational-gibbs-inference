import os
from collections import defaultdict
from typing import Callable, List
import warnings

import numpy as np
import scipy.linalg as sl
import torch

from cdi.util.utils import find_best_model_epoch_from_fs
from cdi.util.stats_utils import load_statistics


def compute_avg_loglikelihood_w_std_err(model: str,
                                        seeds: List[str],
                                        groups: List[str],
                                        base_path: str,
                                        ignore_notfound=False,
                                        filename_suffix=''):
    path_format = (f'{base_path}/{{}}/{model}/{{}}/evaluations/'
                   f'tensors/data_log_probs_test{filename_suffix}.npz')
    out_stats = defaultdict(list)
    for g in groups:
        # Compute average and std err. over different-seeded models
        avg_logliks = []
        for s in seeds:
            # Load evaluation output
            path = path_format.format(g, s)
            if not ignore_notfound:
                data = np.load(path, allow_pickle=True)
            else:
                try:
                    data = np.load(path, allow_pickle=True)
                except FileNotFoundError:
                    warnings.warn(f'File {path} not found, ignoring.')
                    data = {'data_log_probs': np.array([float('nan')])}

            # Compute avg loglikelihood over data
            avg_logliks.append(data['data_log_probs'].mean())
            # avg_logliks.append(data['data_log_probs'])

        # avg_logliks = np.vstack(avg_logliks)
        avg_loglik = np.mean(avg_logliks).item()
        log_lik_std_err = (np.std(avg_logliks, ddof=1)
                           / np.sqrt(len(avg_logliks))).item()
        # log_lik_std_err = (np.std(avg_logliks, ddof=1, axis=0)
        #                    / np.sqrt(len(seeds))).mean().item()

        out_stats['test_log_lik'].append(avg_loglik)
        out_stats['test_log_lik_stderr'].append(log_lik_std_err)
        out_stats['group'].append(g)

    return out_stats


def load_model_params(path, avg_params=False):
    state_dict = torch.load(path, map_location=torch.device('cpu'))['state_dict']

    if 'fa_model.mean' in state_dict:
        return (state_dict['fa_model.mean'].numpy(),
                state_dict['fa_model.log_cov'].numpy(),
                state_dict['fa_model.factor_loadings'].numpy())
    elif avg_params:
        mean = [state_dict[f'fa_model.models.{i}.mean'].numpy()
                for i in range(len(state_dict)//3)]
        log_cov = [state_dict[f'fa_model.models.{i}.log_cov'].numpy()
                   for i in range(len(state_dict)//3)]
        F = [state_dict[f'fa_model.models.{i}.factor_loadings'].numpy()
             for i in range(len(state_dict)//3)]

        return (np.array(mean).mean(axis=0),
                np.array(log_cov).mean(axis=0),
                np.array(F).mean(axis=0))
    else:
        # Load multiple-imp ensemble
        return [(state_dict[f'fa_model.models.{i}.mean'].numpy(),
                 state_dict[f'fa_model.models.{i}.log_cov'].numpy(),
                 state_dict[f'fa_model.models.{i}.factor_loadings'].numpy())
                for i in range(len(state_dict)//3)]


def compute_param_error(model: str,
                        seeds: List[str],
                        groups: List[str],
                        base_path: str,
                        checkpoints='best',
                        mi_avg_params=False):
    path_format = (f'{base_path}/{{}}/{{}}/{{}}/saved_models/'
                   f'{{}}.ckpt')
    out_stats = defaultdict(list)
    for i, g in enumerate(groups):
        # Compute average and std err. over different-seeded models
        all_metrics = defaultdict(list)
        for j, s in enumerate(seeds):
            # Load ground truth params
            truth_path = path_format.format(g, 'fa_ground_truth_cdi', s, 'last')
            mean_truth, log_var_truth, F_truth = load_model_params(truth_path)
            FFT_truth = F_truth @ F_truth.T
            cov_truth = FFT_truth + np.diagflat(np.exp(log_var_truth))

            # Load learnt model params
            model_path = path_format.format(g, model, s, 'last')
            if checkpoints == 'best':
                checkpoint = find_best_model_epoch_from_fs(model_path)
                model_path = path_format.format(g, model, s, f'_ckpt_epoch_{checkpoint}')
            elif isinstance(checkpoints, list):
                checkpoint = checkpoints[i][j]
                model_path = path_format.format(g, model, s, f'_ckpt_epoch_{checkpoint}')
            else:
                model_path = path_format.format(g, model, s, f'{checkpoints}')

            params = load_model_params(model_path, avg_params=mi_avg_params)
            if not isinstance(params, list):
                mean, log_var, F = params
                FFT = F @ F.T
                cov = FFT + np.diagflat(np.exp(log_var))

                d_mean = mean - mean_truth
                d_std = (np.exp((1/2) * log_var)
                         - np.exp((1/2) * log_var_truth))
                cov_truth_sqrt = sl.sqrtm(cov_truth)
                cov_bures = np.sqrt(np.trace(cov_truth + cov
                                             - 2*sl.sqrtm(cov_truth_sqrt
                                                          @ cov
                                                          @ cov_truth_sqrt)))
                wasserstein_d = (np.linalg.norm(mean - mean_truth)**2
                                 + cov_bures**2)

                kl_div = compute_gaussian_kl_div(mean_truth, cov_truth, mean, cov)
                crossent = compute_gaussian_crossentropy(mean_truth, cov_truth, mean, cov)

            else:
                d_mean = []
                d_std = []
                cov_bures = []
                wasserstein_d = []
                kl_div = []
                crossent = []
                for mean, log_var, F in params:
                    FFT = F @ F.T
                    cov = FFT + np.diagflat(np.exp(log_var))

                    d_mean.append(mean - mean_truth)
                    d_std.append(np.exp((1/2) * log_var)
                                 - np.exp((1/2) * log_var_truth))
                    cov_truth_sqrt = sl.sqrtm(cov_truth)
                    cov_bures.append(np.sqrt(np.trace(cov_truth + cov
                                                      - 2*sl.sqrtm(cov_truth_sqrt
                                                                   @ cov
                                                                   @ cov_truth_sqrt))))
                    wasserstein_d.append((np.linalg.norm(mean - mean_truth)**2
                                          + cov_bures[-1]**2))

                    kl_div.append(compute_gaussian_kl_div(mean_truth, cov_truth, mean, cov))
                    crossent.append(compute_gaussian_crossentropy(mean_truth, cov_truth, mean, cov))

                d_mean = np.mean(d_mean)
                d_std = np.mean(d_std)
                cov_bures = np.mean(cov_bures)
                wasserstein_d = np.mean(wasserstein_d)
                kl_div = np.mean(kl_div)
                kl_div = np.mean(kl_div)

            all_metrics['mean_diff'].append(d_mean)
            all_metrics['std_diff'].append(d_std)
            all_metrics['cov_bures'].append(cov_bures)
            all_metrics['wasserstein'].append(wasserstein_d)
            all_metrics['kldiv'].append(kl_div)
            all_metrics['crossent'].append(crossent)

        mean_diff = np.stack(all_metrics['mean_diff'])
        mean_rmse = np.sqrt(np.mean(mean_diff**2, axis=0))
        mean_std_err = (np.std(mean_diff, axis=0, ddof=1)
                        / np.sqrt(len(all_metrics['mean_diff'])).item())
        out_stats['mean_rmse'].append(mean_rmse)
        out_stats['mean_std_err'].append(mean_std_err)

        std_diff = np.stack(all_metrics['std_diff'])
        std_rmse = np.sqrt(np.mean(std_diff**2, axis=0))
        std_std_err = (np.std(all_metrics['std_diff'], axis=0, ddof=1)
                       / np.sqrt(len(all_metrics['std_diff'])).item())
        out_stats['std_rmse'].append(std_rmse)
        out_stats['std_std_err'].append(std_std_err)

        cov_bures_avg = np.mean(all_metrics['cov_bures']).item()
        cov_bures_std_err = (np.std(all_metrics['cov_bures'], ddof=1).item()
                             / np.sqrt(len(all_metrics['cov_bures'])).item())
        out_stats['cov_bures_avg'].append(cov_bures_avg)
        out_stats['cov_bures_std_err'].append(cov_bures_std_err)

        W_avg = np.mean(all_metrics['wasserstein']).item()
        W_std_err = (np.std(all_metrics['wasserstein'], ddof=1).item()
                     / np.sqrt(len(all_metrics['wasserstein'])).item())
        out_stats['W_avg'].append(W_avg)
        out_stats['W_std_err'].append(W_std_err)

        KLD_avg = np.mean(all_metrics['kldiv']).item()
        KLD_std_err = (np.std(all_metrics['kldiv'], ddof=1).item()
                       / np.sqrt(len(all_metrics['kldiv'])).item())
        out_stats['KLD_avg'].append(KLD_avg)
        out_stats['KLD_std_err'].append(KLD_std_err)

        crossent_avg = np.mean(all_metrics['crossent']).item()
        crossent_std_err = (np.std(all_metrics['crossent'], ddof=1).item()
                       / np.sqrt(len(all_metrics['crossent'])).item())
        out_stats['crossent_avg'].append(crossent_avg)
        out_stats['crossent_std_err'].append(crossent_std_err)

        out_stats['group'].append(g)

    out_stats = {
        k: np.stack(v)
        for k, v in out_stats.items()
    }

    return out_stats


def compute_gaussian_crossentropy(mean1, cov1, mean2, cov2):
    return -compute_gaussian_kl_div(mean1, cov1, mean2, cov2) - compute_gaussian_entropy(cov1)


def compute_gaussian_entropy(cov):
    slog_det = np.linalg.slogdet(cov)
    log_det = slog_det[0] * slog_det[1]
    return -1/2*(cov.shape[0]*np.log(2*np.pi*np.exp(1)) + log_det)


def compute_gaussian_kl_div(mean1, cov1, mean2, cov2):
    slog_det1 = np.linalg.slogdet(cov1)
    log_det1 = slog_det1[0] * slog_det1[1]
    slog_det2 = np.linalg.slogdet(cov2)
    log_det2 = slog_det2[0] * slog_det2[1]

    kl_div = (1/2) * (log_det2 - log_det1
                      - mean1.shape[0] + np.trace(np.linalg.inv(cov2) @ cov1)
                      + np.transpose(mean2 - mean1) @ np.linalg.inv(cov2) @ (mean2 - mean1))
    return kl_div


def compute_univariate_gaussian_kl_div(mean1, log_var1, mean2, log_var2):
    kl_divs = (log_var2/2 - log_var1/2 +
               (np.exp(log_var1)
                + (mean1 - mean2)**2)
               / (2*np.exp(log_var2))
               - 1/2)
    return kl_divs


def compute_univariate_gaussian_symmetrised_kl_div(mean1, log_var1, mean2, log_var2):
    kl_divs1 = compute_univariate_gaussian_kl_div(mean1, log_var1, mean2, log_var2)
    kl_divs2 = compute_univariate_gaussian_kl_div(mean2, log_var2, mean1, log_var1)
    return (kl_divs1 + kl_divs2)/2


def compute_wasserstein_distance(mean1, log_var1, mean2, log_var2):
    std1 = np.exp((1/2) * log_var1)
    std2 = np.exp((1/2) * log_var2)

    # B_squared = var1 + var2 - 2*(np.sqrt(var1*var2))
    B_squared = (std1 - std2)**2

    W = (mean1 - mean2)**2 + B_squared
    return W


def load_model_posterior(posterior_key, path):
    # Load data
    data = np.load(path, allow_pickle=True)

    # Get the relevant data
    post_mean = data[f'{posterior_key}_post_mean']
    post_log_var = data[f'{posterior_key}_post_log_var']
    M = data['M']
    return M, post_mean, post_log_var


def compute_avg_posterior_divergence(model1: str,
                                     model2: str,
                                     model1_posterior_key: str,
                                     model2_posterior_key: str,
                                     out_key: str,
                                     seeds: List[str],
                                     groups: List[str],
                                     base_path: str,
                                     divergence_fn: Callable,
                                     divergence_key: str,
                                     return_params=False):
    path_format = (f'{base_path}/{{}}/{{}}/{{}}/evaluations/'
                   f'tensors/posterior_params_test.npz')

    out_stats = defaultdict(list)
    for g in groups:
        # Compute average and std err. over different-seeded models
        all_divs = []
        if return_params:
            m1_post_means = []
            m1_post_logvars = []
            m2_post_means = []
            m2_post_logvars = []
        for s in seeds:
            # Load model1 evaluation output
            path = path_format.format(g, model1, s)
            M1, model1_post_mean, model1_post_log_var = \
                load_model_posterior(model1_posterior_key, path)

            # Load model2 evaluation output
            path = path_format.format(g, model2, s)
            M2, model2_post_mean, model2_post_log_var = \
                load_model_posterior(model2_posterior_key, path)

            # Safety check that we're comparing the same data
            assert np.all(M1 == M2), '`M` matrices do not match!'

            divs = divergence_fn(
                model1_post_mean, model1_post_log_var,
                model2_post_mean, model2_post_log_var
            )

            all_divs.append(divs)
            if return_params:
                m1_post_means.append(model1_post_mean)
                m1_post_logvars.append(model1_post_log_var)
                m2_post_means.append(model2_post_mean)
                m2_post_logvars.append(model2_post_log_var)

        # Stack the outputs
        all_divs = np.concatenate(all_divs)
        out_stats[f'test_{divergence_key}_{out_key}_all'].append(all_divs.T)
        if return_params:
            out_stats[f'test_{out_key}_m1_post_means'].append(np.concatenate(m1_post_means).T)
            out_stats[f'test_{out_key}_m1_post_logvars'].append(np.concatenate(m1_post_logvars).T)
            out_stats[f'test_{out_key}_m2_post_means'].append(np.concatenate(m2_post_means).T)
            out_stats[f'test_{out_key}_m2_post_logvars'].append(np.concatenate(m2_post_logvars).T)

        # Compute average and std. err.
        avg_div = np.mean(all_divs, axis=0)
        divs_std_err = (np.std(all_divs, ddof=1, axis=0)
                        / np.sqrt(len(seeds)))
        # avg_div = np.stack(avg_div)
        # divs_std_err = np.stack(divs_std_err)
        out_stats[f'test_{divergence_key}_{out_key}'].append(avg_div)
        out_stats[f'test_{divergence_key}_std_err_{out_key}'].append(divs_std_err)
        out_stats['group'].append(g)

    out_stats = {
        k: np.stack(v)
        for k, v in out_stats.items()
    }

    return out_stats


def compute_avg_kl_divergence(model1: str,
                              model2: str,
                              model1_posterior_key: str,
                              model2_posterior_key: str,
                              out_key: str,
                              seeds: List[str],
                              groups: List[str],
                              base_path: str,
                              return_params=False):
    return compute_avg_posterior_divergence(
                            model1,
                            model2,
                            model1_posterior_key,
                            model2_posterior_key,
                            out_key,
                            seeds,
                            groups,
                            base_path,
                            divergence_fn=compute_univariate_gaussian_kl_div,
                            divergence_key='avg_kldiv',
                            return_params=return_params)


def compute_avg_symmetrised_kl_divergence(model1: str,
                                          model2: str,
                                          model1_posterior_key: str,
                                          model2_posterior_key: str,
                                          out_key: str,
                                          seeds: List[str],
                                          groups: List[str],
                                          base_path: str,
                                          return_params=False):
    return compute_avg_posterior_divergence(
                            model1,
                            model2,
                            model1_posterior_key,
                            model2_posterior_key,
                            out_key,
                            seeds,
                            groups,
                            base_path,
                            divergence_fn=compute_univariate_gaussian_symmetrised_kl_div,
                            divergence_key='avg_sym_kldiv',
                            return_params=return_params)


def compute_avg_wasserstein_distance(model1: str,
                                     model2: str,
                                     model1_posterior_key: str,
                                     model2_posterior_key: str,
                                     out_key: str,
                                     seeds: List[str],
                                     groups: List[str],
                                     base_path: str,
                                     return_params=False):
    return compute_avg_posterior_divergence(
                            model1,
                            model2,
                            model1_posterior_key,
                            model2_posterior_key,
                            out_key,
                            seeds,
                            groups,
                            base_path,
                            divergence_fn=compute_wasserstein_distance,
                            divergence_key='avg_wass_dist',
                            return_params=return_params)


def compute_avg_num_var_updates_per_dim(model: str,
                                        seeds: List[str],
                                        groups: List[str],
                                        base_path: str):
    path_format = (f'{base_path}/{{}}/{{}}/{{}}/logs/'
                   f'tensors/num_var_updates_final.npz')

    out_stats = defaultdict(list)
    for g in groups:
        # Compute average over different-seeded models
        all_updates = []
        for s in seeds:
            # Load model output
            path = path_format.format(g, model, s)
            data = np.load(path, allow_pickle=True)

            # Get the relevant data
            num_updates = data['num_updates']
            # curr_epoch = data['curr_epoch']

            all_updates.append(num_updates)

        # Stack the outputs
        all_updates = np.concatenate(all_updates)

        # Compute average
        all_updates = np.mean(all_updates, axis=0)

        out_stats[f'num_var_updates'].append(all_updates)
        out_stats['group'].append(g)

    out_stats = {
        k: np.stack(v)
        for k, v in out_stats.items()
    }

    return out_stats


def compute_avg_test_loglikelihood_vs_train_time(orig_model: str,
                                                 refit_model: str,
                                                 ckpts: List[int],
                                                 seeds: List[str],
                                                 group: str,
                                                 base_path: str):
    summary_format = (f'{base_path}/{group}/{{}}/{{}}/logs/summary.csv')
    path_format = (f'{base_path}/{group}/{{}}/{{}}/evaluations/'
                   f'tensors/data_log_probs_test.npz')
    out_stats = defaultdict(list)
    for ckpt in ckpts:
        _ckpt = f'ckpt_{ckpt}_cust' if ckpt != -1 else 'ckpt_last'
        model = f'{refit_model}_{_ckpt}'

        # Compute average and std err. over different-seeded models
        avg_logliks = []
        train_times = []
        # train_imp_times = []
        for s in seeds:
            # Load evaluation output
            path = path_format.format(model, s)
            data = np.load(path, allow_pickle=True)

            # Compute avg loglikelihood over data
            avg_logliks.append(data['data_log_probs'].mean())
            # avg_logliks.append(data['data_log_probs'])

            # Load summary data and add training data
            orig_path = summary_format.format(orig_model, s)
            stat = load_statistics(os.path.dirname(orig_path), os.path.basename(orig_path))
            train_times.append(np.sum(stat['train_time'][:ckpt]))
            # train_imp_times.append(np.sum(stat['train_imp_time'][:ckpt]))

        # avg_logliks = np.vstack(avg_logliks)
        avg_loglik = np.mean(avg_logliks).item()
        log_lik_std_err = (np.std(avg_logliks, ddof=1)
                           / np.sqrt(len(avg_logliks))).item()
        # log_lik_std_err = (np.std(avg_logliks, ddof=1, axis=0)
        #                    / np.sqrt(len(seeds))).mean().item()
        train_time = np.mean(train_times)
        # train_imp_time = np.mean(train_imp_times)

        out_stats['test_log_lik'].append(avg_loglik)
        out_stats['test_log_lik_stderr'].append(log_lik_std_err)
        out_stats['train_time'].append(train_time)
        # out_stats['train_imp_time'].append(train_imp_time)
        ckpt = ckpt if ckpt != -1 else stat['curr_epoch'][-1]
        out_stats['epoch'].append(ckpt)
        out_stats['train_time_per_epoch'].append(train_time/ckpt)

    return out_stats


def compute_avg_test_loglikelihood_vs_train_time_flows(model: str,
                                                     ckpts: List[int],
                                                     seeds: List[str],
                                                     group: str,
                                                     base_path: str,
                                                     ignore_notfound=False,
                                                     no_suffix=False):
    summary_format = (f'{base_path}/{group}/{{}}/{{}}/logs/summary.csv')
    if no_suffix:
        path_format = (f'{base_path}/{group}/{{}}/{{}}/evaluations/'
                    'tensors/data_log_probs_test.npz')
        assert len(ckpts) == 1
    else:
        path_format = (f'{base_path}/{group}/{{}}/{{}}/evaluations/'
                    'tensors/data_log_probs_test_{}.npz')

    out_stats = defaultdict(list)
    for ckpt in ckpts:
        _ckpt = f'{ckpt}_cust' if ckpt != -1 else 'last'

        # Compute average and std err. over different-seeded models
        avg_logliks = []
        train_times = []
        for s in seeds:
            # Load evaluation output
            if no_suffix:
                path = path_format.format(model, s)
            else:
                path = path_format.format(model, s, _ckpt)
            try:
                data = np.load(path, allow_pickle=True)
            except FileNotFoundError as e:
                if ignore_notfound:
                    continue
                else:
                    raise e

            # Compute avg loglikelihood over data
            avg_logliks.append(data['data_log_probs'].mean())

            # Load summary data and add training data
            stat_path = summary_format.format(model, s)
            stat = load_statistics(os.path.dirname(stat_path), os.path.basename(stat_path))
            train_times.append(np.sum(stat['train_time'][:ckpt]))

        avg_loglik = np.mean(avg_logliks).item()
        log_lik_std_err = (np.std(avg_logliks, ddof=1)
                           / np.sqrt(len(avg_logliks))).item()
        train_time = np.mean(train_times)

        out_stats['test_log_lik'].append(avg_loglik)
        out_stats['test_log_lik_stderr'].append(log_lik_std_err)
        out_stats['train_time'].append(train_time)
        ckpt = ckpt if ckpt != -1 else stat['curr_epoch'][-1]
        out_stats['epoch'].append(ckpt)
        out_stats['train_time_per_epoch'].append(train_time/ckpt)

    return out_stats
