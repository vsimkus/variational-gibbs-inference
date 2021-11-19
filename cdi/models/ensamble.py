import torch
from pytorch_lightning import LightningModule


class Ensamble(LightningModule):
    """
    Used to fit separate model for each imputed dataset, when using
    multiple imputation (e.g. MICE)
    """
    def __init__(self, hparams, model_class):
        super().__init__()
        self.hparams = hparams

        # assert

        models = []
        for i in range(max(self.hparams.data.num_imputed_copies)):
            models.append(model_class(self.hparams))

        self.models = torch.nn.ModuleList(models)

    def forward(self, X, M):
        if self.training:
            X = X.reshape(-1, len(self.models), X.shape[-1])
            M = M.reshape(-1, len(self.models), M.shape[-1])
        else:
            X = X.unsqueeze(1).expand(-1, len(self.models), -1)
            M = M.unsqueeze(1).expand(-1, len(self.models), -1)

        out = torch.zeros(X.shape[0], len(self.models), device=X.device)
        for i, model in enumerate(self.models):
            out[:, i] = self.models[i](X[:, i], M[:, i])

        return out.mean(dim=-1)

    def reset_parameters(self):
        for model in self.models:
            model.reset_parameters()

    def freeze_decoder(self):
        for model in self.models:
            model.freeze_decoder()

    def freeze_encoder(self):
        for model in self.models:
            model.freeze_encoder()

    def on_epoch_start(self):
        for model in self.models:
            model.on_epoch_start()

    @property
    def cum_batch_size_called(self):
        cum_batch_size_called = 0
        for model in self.models:
            cum_batch_size_called += model.cum_batch_size_called

        return cum_batch_size_called
