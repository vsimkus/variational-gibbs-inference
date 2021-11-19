from collections import OrderedDict

import torch
import numpy as np

import cdi.trainers.complete_mle as cm


class MoGEM(cm.CompleteMLE):
    """
    Expectation-Maximisation estimation for Mixture of Gaussians model.
    """
    def __init__(self, hparams, model=None):
        super().__init__(hparams, model=model)

        if self.num_imputed_copies_scheduler is not None:
            num_imputed_copies = self.num_imputed_copies_scheduler.get_value()
        elif isinstance(self.hparams.data.num_imputed_copies, list):
            num_imputed_copies = self.hparams.data.num_imputed_copies[0]
        else:  # BC
            num_imputed_copies = self.hparams.data.num_imputed_copies

        assert num_imputed_copies == 1,\
            'EM shouldn\'t use augmented data!'

    # Setup

    def configure_optimizers(self):
        return None

    def setup(self, stage):
        out = super().setup(stage)

        # Make sure that the batch size is larger than the data
        if stage == 'fit':
            assert len(self.train_dataset) <= self.hparams.data.batch_size, \
                'For EM the batch size must be >= to the size of the dataset!'
            # assert len(self.val_dataset) <= self.hparams.data.batch_size, \
            #     'For EM the batch size must be >= to the size of the dataset!'

        return out

    # Training

    def backward(self, use_amp, loss, optimizer, optimizer_idx):
        # Do nothing on backward for EM (or could set the updated values here
        # instead of the training loop)
        pass

    def compute_updated_parameters_with_missing(self, batch):
        X, M = batch[:2]
        M_not = (~M).unsqueeze(1)
        M = M.unsqueeze(1)

        # Model params
        covs = self.fa_model.covs.unsqueeze(0)
        means = self.fa_model.means.unsqueeze(0)
        comp_logits = self.fa_model.comp_logits
        comp_probs = torch.softmax(comp_logits, dim=0)

        sigma_mm = covs * M_not.unsqueeze(-1) * M_not.unsqueeze(-2)
        sigma_mo = covs * M_not.unsqueeze(-1) * M.unsqueeze(-2)
        sigma_oo = covs * M.unsqueeze(-1) * M.unsqueeze(-2)

        M_eye = torch.eye(X.shape[-1]).unsqueeze(0).unsqueeze(0) * M_not.unsqueeze(-1)
        sigma_oo_with_ones_in_missing_diag_entries = sigma_oo + M_eye
        sigma_oo_inv = torch.inverse(sigma_oo_with_ones_in_missing_diag_entries) - M_eye

        sigma_mo_oo_inv = sigma_mo @ sigma_oo_inv

        xo_minus_mean = (X.unsqueeze(1) - means)*M
        mean_x = means*M_not + (sigma_mo_oo_inv @ xo_minus_mean.unsqueeze(-1)).squeeze(-1)

        cov_x = sigma_mm - sigma_mo_oo_inv @ sigma_mo.transpose(-1, -2)

        D = M.sum(-1)
        cond_log_probs = 0.5*(-D * torch.log(torch.tensor(2*np.pi))
                    -torch.logdet(sigma_oo_with_ones_in_missing_diag_entries)
                    -(xo_minus_mean.unsqueeze(-2) @ sigma_oo_inv @ xo_minus_mean.unsqueeze(-1)).squeeze()
        )
        comp_log_probs = torch.log(comp_probs)
        joint_probs = torch.exp(comp_log_probs + cond_log_probs)
        norm = joint_probs.sum(-1)
        cond_probs = joint_probs / norm.unsqueeze(-1)

        responsibilities = cond_probs.sum(0)

        # import pdb; pdb.set_trace()

        comp_probs_up = responsibilities/X.shape[0]
        comp_logits_up = torch.log(comp_probs_up)

        X = X.unsqueeze(1)*M + mean_x*M_not

        means_up = (X * cond_probs.unsqueeze(-1)).sum(0)/responsibilities.unsqueeze(-1)

        dif = (X - means_up)
        covs_up = (((dif.unsqueeze(-1) @ dif.unsqueeze(-2)) + cov_x)*cond_probs.unsqueeze(-1).unsqueeze(-1)).sum(0) / responsibilities.unsqueeze(-1).unsqueeze(-1)
        # Ensure positive definiteness by small value
        covs_up += torch.eye(covs.shape[0]).unsqueeze(0)*1e-8
        Ls_up = torch.cholesky(covs_up, upper=False)

        return means_up, Ls_up, comp_logits_up

    def training_step(self, batch, batch_idx):
        # Gradients not needed for EM
        with torch.no_grad():
            means_up, Ls_up, comp_logits_up = self.compute_updated_parameters_with_missing(batch)

            # Update values
            self.fa_model.means.data = means_up
            self.fa_model.scale_tril_entries.data = Ls_up[:, self.fa_model.tril_indices[0], self.fa_model.tril_indices[1]]
            self.fa_model.unconstrained_diags.data = torch.log(Ls_up[:, self.fa_model.diag_indices, self.fa_model.diag_indices])
            self.fa_model.comp_logits.data = comp_logits_up

            # Compute the observed-data log-probability
            # X, M = batch[:2]
            # log_probs = self.fa_model(X, M)
            # log_prob = log_probs.mean()

        pbar = {
            # 'train_log_lik': log_prob.item()
            # 'train_entropy': entropy.item(),
        }
        output = OrderedDict({
            'loss': torch.tensor(0.),
            'progress_bar': pbar,
        })
        return output

    # Validation

    def validation_step(self, batch, batch_idx):
        X, M = batch[:2]

        # Compute the observed-data log-probability
        # with torch.no_grad():
            # log_probs = self.fa_model.log_marginal_prob(X, M)
            # log_probs = self.fa_model(X, M)
            # log_prob = log_probs.mean()

        output = OrderedDict({
            'val_loss': torch.tensor(0.),
            # 'val_log_lik': log_prob.item()
        })
        return output
