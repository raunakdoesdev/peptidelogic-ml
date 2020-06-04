import copy

import numpy as np
import optuna
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from sklearn import metrics


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
        self.lightning_module = None
        self.trainer = None

    def train_model(self, max_epochs=500, trial=None):
        import os
        if os.name == 'nt':
            self.trainer = Trainer(max_epochs=max_epochs)
        else:
            self.trainer = Trainer(gpus=1, max_epochs=max_epochs)
        hparams = HParams(self.params, trial)
        self.lightning_module = ItchDetector(self.model(hparams), hparams, self.dataset, trial)
        self.tune_lr(self.trainer)
        self.trainer.fit(self.lightning_module)
        self.lightning_module.model.load_state_dict(self.lightning_module.best_state_dic)  # grab best model
        return self.lightning_module, self.lightning_module.max_auc

    def get_model(self, state_dict):
        hparams = HParams(self.params)
        self.lightning_module = ItchDetector(self.model(hparams), hparams, self.dataset, trial=None)
        self.lightning_module.model.load_state_dict(state_dict)  # grab best model
        return self.lightning_module

    def objective(self, trial):
        return self.train_model(trial=trial)[1]

    def hyperparemeter_optimization(self, n_trials=None, timeout=None):
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

        return trial.value

    def tune_lr(self, trainer):
        lr_finder = trainer.lr_find(self.lightning_module)
        new_lr = lr_finder.suggestion()
        self.lightning_module.hparams.learning_rate = new_lr


class ItchDetector(pl.LightningModule):
    def __init__(self, model, hparams, dataset, trial=None):
        super().__init__()
        if hparams.percent_data is None:
            self.test_set, self.val_set, self.train_set = dataset.split_dataset([0.15, 0.15, 0.7])
        else:
            self.test_set, self.val_set, _, self.train_set = dataset.split_dataset(
                [0.15, 0.15, 0.7 * (1 - hparams.percent_data), 0.7 * hparams.percent_data])

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

    def train_dataloader(self):
        return self.train_set

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        weight = (self.hparams.weight * (y == 1) + (y == 0)).float()
        y_true = y.contiguous().view(-1).cpu()
        y_pred = y_hat.contiguous().view(-1).cpu()
        return {'loss': F.binary_cross_entropy(y_hat, y, weight=weight),
                'auc': metrics.average_precision_score(y_true, y_pred)}

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

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        weight = (self.hparams.weight * (y == 1) + (y == 0)).float()

        y_true = y.contiguous().view(-1).cpu()
        y_pred = y_hat.contiguous().view(-1).cpu()
        auc = metrics.roc_auc_score(y_true, y_pred)

        thresholds = [0.2, 0.5, 0.7, 0.9]
        specificity = []
        sensitivity = []
        precision = []
        for threshold in thresholds:
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred > threshold).ravel()
            sensitivity.append(tp / (tp + fn))
            specificity.append(tn / (tn + fp))
            precision.append(tp / (tp + fp))

        return {'loss': F.binary_cross_entropy(y_hat, y, weight=weight),
                'auc': np.array(auc), 'specificity': np.array(specificity), 'sensitivity': np.array(sensitivity),
                'average_precision': metrics.average_precision_score(y_true, y_pred),
                'precision': np.array(precision),
                'thresholds': np.array(thresholds)}

    def test_dataloader(self):
        return self.test_set

    def test_epoch_end(self, outputs):
        log = {}
        for key in outputs[0].keys():
            log[key] = outputs[0][key]
        self.test_log = log
        return log

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
