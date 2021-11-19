from collections import OrderedDict

import torch

import cdi.trainers.complete_mle as cm


class GaussEM(cm.CompleteMLE):
    """
    Expectation-Maximisation estimation for Gaussian model.
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
        M_not = ~M

        # Model params
        cov = self.fa_model.cov
        mean = self.fa_model.mean

        sigma_mm = cov * M_not.unsqueeze(-1) * M_not.unsqueeze(-2)
        sigma_mo = cov * M_not.unsqueeze(-1) * M.unsqueeze(-2)
        sigma_oo = cov * M.unsqueeze(-1) * M.unsqueeze(-2)

        M_eye = torch.eye(X.shape[-1]).unsqueeze(0) * M_not.unsqueeze(-1)
        sigma_oo_inv = torch.inverse(sigma_oo + M_eye) - M_eye

        sigma_mo_oo_inv = sigma_mo @ sigma_oo_inv

        xo_minus_mean = (X - mean)*M
        mean_x = mean*M_not + (sigma_mo_oo_inv @ xo_minus_mean.unsqueeze(-1)).squeeze(-1)

        cov_x = sigma_mm - sigma_mo_oo_inv @ sigma_mo.transpose(-1, -2)

        X = X*M + mean_x*M_not

        mean_up = X.mean(dim=0)

        dif = (X - mean_up)
        cov_up = ((dif.unsqueeze(-1) @ dif.unsqueeze(-2)) + cov_x).mean(dim=0)
        L_up = torch.cholesky(cov_up, upper=False)

        return mean_up, L_up

    def training_step(self, batch, batch_idx):
        # Gradients not needed for EM
        with torch.no_grad():
            mean_up, L_up = self.compute_updated_parameters_with_missing(batch)

            # Update values
            self.fa_model.mean.data = mean_up
            self.fa_model.scale_tril_entries.data = L_up[self.fa_model.tril_indices[0], self.fa_model.tril_indices[1]]
            self.fa_model.unconstrained_diag.data = torch.log(L_up[self.fa_model.diag_indices, self.fa_model.diag_indices])

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
