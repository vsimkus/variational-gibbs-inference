import time
from collections import OrderedDict

import torch

from cdi.trainers.variational_cdi import VarCDI
from cdi.util.multi_optimiser import MultiOptimiser


class VarCDIEM(VarCDI):

    def configure_optimizers(self):
        optimiser = super().configure_optimizers()

        # Get the FA and Var. optimisers
        model_opt = MultiOptimiser(
                    fa_model_opt=optimiser.optimisers['fa_model_opt'])
        var_opt = optimiser.optimisers['var_model_opt']

        if 'mis_model_opt' in optimiser.optimisers:
            model_opt.add_optimisers(
                    mis_model_opt=optimiser.optimisers['mis_model_opt'])

        self.model_opt = model_opt
        return [model_opt, var_opt]

    def forward(self, X, M):
        if self.training:
            self.model_opt.add_run_opt('fa_model_opt')
            if hasattr(self, 'mis_model'):
                self.model_opt.add_run_opt('mis_model_opt')

        return super().forward(X, M)

    def select_one_chain(self, batch):
        if self.num_imputed_copies_scheduler is not None:
            num_imputed_copies = self.num_imputed_copies_scheduler.get_value()
        elif isinstance(self.hparams.data.num_imputed_copies, list):
            num_imputed_copies = self.hparams.data.num_imputed_copies[0]
        else:  # BC
            num_imputed_copies = self.hparams.data.num_imputed_copies

        X, M, I, OI, incomp_mask = batch
        full_mask = ~incomp_mask
        incomp_idx = torch.where(incomp_mask)[0]

        # Select only one chain for each incomplete sample
        which_chain = torch.randint(low=0,
                                    high=num_imputed_copies,
                                    size=(1,))
        selected_chain_idx = torch.arange(
                                    start=which_chain.item(),
                                    end=incomp_idx.shape[0],
                                    step=num_imputed_copies)
        selected_chain_idx = incomp_idx[selected_chain_idx]
        selected_incomp_mask = torch.zeros_like(incomp_mask)
        selected_incomp_mask[selected_chain_idx] = 1

        # Select all fully-observed and only the few selected incomplete
        selected = selected_incomp_mask + full_mask

        X = X[selected, :]
        M = M[selected, :]
        I = I[selected]
        OI = OI[selected]
        incomp_mask = incomp_mask[selected]

        return (X, M, I, OI, incomp_mask)

    def maximise_model_step(self, batch):
        X, M, I, OI, incomp_mask = batch

        if self.num_imputed_copies_scheduler is not None:
            num_imputed_copies = self.num_imputed_copies_scheduler.get_value()
        elif isinstance(self.hparams.data.num_imputed_copies, list):
            num_imputed_copies = self.hparams.data.num_imputed_copies[0]
        else:  # BC
            num_imputed_copies = self.hparams.data.num_imputed_copies

        fa_log_prob, mis_log_prob = self.forward(X, M)
        if mis_log_prob is not None:
            log_probs = fa_log_prob + mis_log_prob
        else:
            log_probs = fa_log_prob
        if num_imputed_copies > 1:
            unique_oi, oi_inv_idx, oi_counts = torch.unique(
                                                    OI,
                                                    return_inverse=True,
                                                    return_counts=True)
            N_true = unique_oi.shape[0]

            log_probs /= oi_counts[oi_inv_idx]
            log_prob = log_probs.sum() / N_true
        else:
            log_prob = log_probs.mean()
        pbar = {
            'train_M_step_log_lik': log_prob.item()
        }
        output = OrderedDict({
            'loss': -log_prob,
            'progress_bar': pbar,
        })
        pbar['train_fa_log_prob'] = fa_log_prob.mean().item()
        if mis_log_prob is not None:
            pbar['train_mis_log_prob'] = mis_log_prob.mean().item()
        return output

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        Performs CDI update and imputation of missing values.
        """
        if optimizer_idx == 0:
            # Imputation (E-step)
            imp_start_time = time.time()
            logs = self.impute_batch(batch)
            imp_time = time.time() - imp_start_time

            # Update FA model (M-step)
            output = self.maximise_model_step(batch)
            output['progress_bar'].update(logs)
            output['progress_bar']['imp_time'] = imp_time

        if optimizer_idx == 1:
            # Update Var. model (M-step 2)
            # one_chain_batch = self.select_one_chain(batch)
            # output = self.update_step(one_chain_batch, only_var=True)
            output = self.update_step(batch, only_var=True)
        return output
