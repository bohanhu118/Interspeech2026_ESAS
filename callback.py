import lightning.pytorch as pl
import torch
import torchinfo
from lightning.pytorch.callbacks import BasePredictionWriter
import lightning as L
import numpy as np
from typing import Dict, Any
from lightning.pytorch.utilities.model_summary import ModelSummary


class OverrideEpochStepCallback(pl.callbacks.Callback):
    """
    Override the step axis in Tensorboard with epoch. Just ignore the warning message popped out.
    """
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.log("step", trainer.current_epoch)


class PrintAndLogModelProfile(pl.callbacks.Callback):
    @staticmethod
    def generate_model_profile(model, example_input_size):
        # Summary the model profile
        print("\n Model Profile:")
        model_profile = torchinfo.summary(model, input_size=example_input_size)
        macc = model_profile.total_mult_adds
        params = model_profile.total_params
        print('MACC:\t \t %.6f' % (macc / 1e6), 'M')
        print('Params:\t \t %.3f' % (params / 1e3), 'K\n')
        # Convert the summary to string
        model_summary = str(model_profile)
        model_summary += f'\n MACC:\t \t {macc / 1e6:.3f}M'
        model_summary += f'\n Params:\t \t {params / 1e3:.3f}K\n'
        model_summary = model_summary.replace('\n', '<br/>').replace(' ', '&nbsp;').replace('\t', '&emsp;')
        return model_summary

    @staticmethod
    def get_example_input_size(trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Get example input directly from dataset
        if trainer.state.stage == "train":
            dataset = trainer.datamodule.train_set
        else:
            dataset = trainer.datamodule.test_set

        example = dataset[0]  # dataset must return (x, y) or dict
        if isinstance(example, dict):
            x = example["wav"].unsqueeze(0)
        elif isinstance(example, (list, tuple)):
            x = example[0].unsqueeze(0)
        else:
            raise TypeError(f"Unsupported dataset return type: {type(example)}")

        x = pl_module.spec_extractor(x).unsqueeze(1) if pl_module.spec_extractor is not None else x.unsqueeze(1)
        return x.size()

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Get example_input_size for generating model profile
        example_input_size = self.get_example_input_size(trainer, pl_module)
        # Get tensorboard
        tensorboard = pl_module.logger.experiment
        model_summary = self.generate_model_profile(pl_module.backbone, example_input_size)
        tensorboard.add_text('model_profile', model_summary)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Get example_input_size for generating model profile
        example_input_size = self.get_example_input_size(trainer, pl_module)
        # Get tensorboard
        tensorboard = pl_module.logger.experiment
        model_summary = self.generate_model_profile(pl_module.backbone, example_input_size)
        tensorboard.add_text('model_profile', model_summary)


class LoadCheckpointAndFreezeBackbone(pl.callbacks.Callback):
    """
    Load pre-trained checkpoint to backbone and freeze the backbone.
    """
    def __init__(self, ckpt_path) -> None:
        super().__init__()
        self.ckpt_path = ckpt_path

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        prefix = 'backbone.'
        # Load state dict from `ckpt_path`
        ckpt = torch.load(self.ckpt_path)
        state_dict = ckpt['state_dict']
        updated_state_dict = {key.replace(prefix, ''): value for key, value in state_dict.items()}
        pl_module.backbone.load_state_dict(updated_state_dict, strict=True)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Freeze all parameters
        for param in pl_module.backbone.parameters():
            param.requires_grad = False
        pl_module.backbone.eval()


class PredictionWriter(BasePredictionWriter):
    """
    Write the predictions of a pretrained model into a pt file.
    """
    def __init__(self, predict_subset, label="scene", write_interval="epoch"):
        super().__init__(write_interval)
        self.predict_subset = predict_subset
        self.label = label

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        pass

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        save_path = trainer.ckpt_path.split("checkpoints")[0]
        preds = torch.cat(predictions, dim=0)
        torch.save(preds, f"{save_path}predictions_{self.predict_subset}.pt")
        print(f"\nSuccessfully save predictions to {save_path}predictions_{self.label}_{self.predict_subset}.pt")