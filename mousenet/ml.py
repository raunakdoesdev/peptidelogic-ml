import copy

import numpy as np
import optuna
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer


class HParams:
    def __init__(self, parameters, trial=None):
        for key, value in parameters.items():
            if type(value) is tuple:
                value = value[0] if trial is None else trial.suggest_int(key, *value[1])
            setattr(self, key, value)

        self.learning_rate = 0.01


class Runner:
    def __init__(self, model, params, dataset):
        self.model = model
        self.params = params
        self.dataset = dataset

    def train_model(self, max_epochs=500, trial=None):
        import os
        if os.name == 'nt':
            trainer = Trainer(max_epochs=max_epochs)
        else:
            trainer = Trainer(gpus=1, max_epochs=max_epochs)
        hparams = HParams(self.params, trial)
        lightning_module = ItchDetector(self.model(hparams), hparams, self.dataset, trial)
        self.tune_lr(trainer, lightning_module)
        trainer.fit(lightning_module)
        lightning_module.model.load_state_dict(lightning_module.best_state_dic)  # grab best model
        return lightning_module, lightning_module.max_auc

    def objective(self, trial):
        return self.train_model(trial=trial)[1]

    def hyperparemeter_optimization(self, n_trials=1000, timeout=600):
        pruner = optuna.pruners.SuccessiveHalvingPruner()
        study = optuna.create_study(direction="maximize", pruner=pruner)
        try:
            study.optimize(self.objective, n_trials=n_trials, timeout=timeout)
        except KeyboardInterrupt:
            pass

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    def tune_lr(self, trainer, lightning_module):
        lr_finder = trainer.lr_find(lightning_module)
        new_lr = lr_finder.suggestion()
        lightning_module.hparams.learning_rate = new_lr


class ItchDetector(pl.LightningModule):
    def __init__(self, model, hparams, dataset, trial=None):
        super().__init__()
        self.train_set, self.val_set = dataset.split_dataset(0.75)
        self.train_set, _ = self.train_set.split_dataset(hparams.train_val_split)

        print(f'Training Set Size: {self.train_set[0][0].shape}')
        print(f'Validation Set Size: {self.val_set[0][0].shape}')

        self.hparams = hparams
        self.max_auc = None
        self.cur_auc = None
        self.best_state_dic = None
        self.model = model
        self.trial = trial

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        weight = self.hparams.weight * (y == 1) + (y == 0).float()
        return {'loss': F.binary_cross_entropy(y_hat, y, weight=weight)}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        weight = (self.hparams.weight * (y == 1) + (y == 0)).float()

        tpr = [0]
        fpr = [0]
        epsilon = 0.1
        thresh = 1
        auc = 0
        while thresh >= 0:
            pred_y = y_hat >= thresh
            tpr.append(float(torch.sum((y == pred_y) * (y == 1))) / float(torch.sum(y)))
            fpr.append(float(torch.sum((y != pred_y) * (y == 0))) / float(torch.sum(y == 0)))
            auc += (fpr[-1] - fpr[-2]) * (tpr[-2] + tpr[-1]) / 2
            thresh -= epsilon
        return {'loss': F.binary_cross_entropy(y_hat, y, weight=weight),
                'auc': auc}

    def train_dataloader(self):
        return self.train_set

    def val_dataloader(self):
        return self.val_set

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_auc = np.mean([x['auc'] for x in outputs])
        log = {'val_loss': avg_loss, 'val_auc': avg_auc}
        log['log'] = copy.deepcopy(log)

        if self.max_auc is None or self.max_auc < avg_auc:
            self.max_auc = avg_auc
            self.best_state_dic = self.model.state_dict()

        self.cur_auc = avg_auc

        if self.trial is not None:
            self.trial.report(avg_auc, step=self.current_epoch)
            if self.trial.should_prune():
                message = "Trial was pruned at epoch {}.".format(self.current_epoch)
                raise optuna.exceptions.TrialPruned(message)
        return log

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
