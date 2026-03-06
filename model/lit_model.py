from typing import Dict, Optional
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import lightning as L
import hydra
import torch.nn.functional as F
from model.backbone import _BaseBackbone
from model.classifier import build_new_classifier_from_old
from model.shared import ReverseLayerF
from util.data_augmentation import _DataAugmentation
from util.spec_extractor import _SpecExtractor
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from util.lr_scheduler import exp_warmup_linear_down


class LitAcousticClassificationSystem(L.LightningModule):
    """
    Acoustic Classification system based on LightningModule.
    Backbone model, data augmentation techniques and spectrogram extractor are designed to be plug-and-played.
    Backbone architecture, system complexity, classification report and confusion matrix are shown at test stage.

    Args:
        backbone (_BaseBackbone): Deep neural network backbone, e.g. cnn, transformer...
        data_augmentation (dict): A dictionary containing instances of data augmentation techniques in util/. Options: MixUp, FreqMixStyle, DeviceImpulseResponseAugmentation, SpecAugmentation. Set each to ``None`` if not use one of them.
        class_label (str): Class label. e.g. scene.
        spec_extractor (_SpecExtractor): Spectrogram extractor used to transform 1D waveforms to 2D spectrogram. If ``None``, the input features should be 2D spectrogram.
    """
    def __init__(self,
                 backbone: _BaseBackbone,
                 data_augmentation: Dict[str, _DataAugmentation] = None,
                 class_label: str = 'scene',
                 spec_extractor: _SpecExtractor = None,
                 ):
        super(LitAcousticClassificationSystem, self).__init__()
        self.backbone = backbone
        self.data_aug = SimpleNamespace(**data_augmentation) if data_augmentation else SimpleNamespace()
        self.class_label = class_label
        self.spec_extractor = spec_extractor

        # Override self.mixup_label_keys in subclasses when applying more labels during training,
        # e.g. soft scene labels for knowledge distillation or soft event labels for adversarial training
        self.mixup_label_keys = [self.class_label]

        # Save the hyperparameters for loading the model from a checkpoint.
        self.save_hyperparameters()

    def load_state_dict(self, state_dict, strict: bool = False):
        # force strict=False so extra keys are ignored
        return super().load_state_dict(state_dict, strict=strict)

    def _apply_data_aug(self, x, labels):
        """Shared preprocessing pipeline."""
        # Get the sampling rate
        sr = self.trainer.train_dataloader.dataset.sr

        # DIR augmentation
        if hasattr(self.data_aug, 'dir_aug'):
            x = self.data_aug.dir_aug(x, labels['device'], sr)

        # Spectrogram extraction
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor else x.unsqueeze(1)

        # Fre-MixStyle
        if hasattr(self.data_aug, 'mix_style'):
            x = self.data_aug.mix_style(x)

        # Spec augmentations
        if hasattr(self.data_aug, 'spec_aug'):
            x = self.data_aug.spec_aug(x)

        # MixUp
        if hasattr(self.data_aug, 'mix_up'):
            # Separate labels for mixup vs fixed
            mix_labels = [labels[k] for k in self.mixup_label_keys]
            x_mixed, labels_mixed = self.data_aug.mix_up([x], mix_labels)
            x = x_mixed[0]

            # Update only mixup_label_keys
            for k, v in zip(self.mixup_label_keys, labels_mixed):
                labels[k] = v

        return x, labels

    def preprocess_batch(self, batch_data_dict):
        # Load waveform
        x = batch_data_dict['wav']

        # Collect labels into dict
        labels = {k: batch_data_dict[k] for k in batch_data_dict if not k.endswith('wav')}

        # Apply data augementation pipeline
        x, labels = self._apply_data_aug(x, labels)
        return x, labels

    @staticmethod
    def accuracy(logits, labels, return_pred=False):
        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == labels).item() / len(labels)
        if return_pred:
            return acc, pred
        else:
            return acc

    def compute_training_metrics(self, y_hat, y):
        if hasattr(self.data_aug, 'mix_up'):
            train_acc = self.data_aug.mix_up.cal_accuracy(y_hat, y)
            train_loss = self.data_aug.mix_up.cal_loss(y_hat, y, F.cross_entropy)
        else:
            train_acc = self.accuracy(y_hat, y)
            train_loss = F.cross_entropy(y_hat, y)
        self.log_dict({f'train_acc_{self.class_label}': train_acc, f'train_loss_{self.class_label}': train_loss},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def forward(self, x, return_emb=False):

        if torch.backends.mps.is_available():
            device = torch.device("mps")
            # print("Device set to: mps")
        else:
            device = torch.device("cpu")
            # print("Device set to: cpu")

        self.backbone.to(device)
        y_hat, emb = self.backbone(x)
        if return_emb:
            return y_hat, emb
        else:
            return y_hat

    def training_step(self, batch_data_dict, batch_idx):
        # Pre-processing: extract audio spectrogram and apply data augmentation
        x, labels = self.preprocess_batch(batch_data_dict)
        # Get the ground trues and predictions
        y = labels[self.class_label]
        y_hat = self(x)
        # Calculate the loss and accuracy
        train_loss = self.compute_training_metrics(y_hat, y)
        return train_loss

    def validation_step(self, batch_data_dict):
        x, y = batch_data_dict['wav'], batch_data_dict[self.class_label]
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        val_acc = self.accuracy(y_hat, y)
        self.log_dict({f'val_loss_{self.class_label}': val_loss, f'val_acc_{self.class_label}': val_acc},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch_data_dict):
        x, y = batch_data_dict['wav'], batch_data_dict[self.class_label]
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)
        y_hat, emb = self(x, return_emb=True)
        test_loss = F.cross_entropy(y_hat, y)
        test_acc, pred = self.accuracy(y_hat, y, return_pred=True)
        self.log_dict({f'test_loss_{self.class_label}': test_loss, f'test_acc_{self.class_label}': test_acc})
        # Return `ground_true`, `prediction` and `domain_label` for result analysis callback
        if "domain" in list(batch_data_dict.keys()):
            d = batch_data_dict['domain']
            return {'y': y, 'y_hat': y_hat, 'pred': pred, 'd': d, 'emb': emb}
        else:
            return {'y': y, 'y_hat': y_hat, 'pred': pred, 'emb': emb}

    def predict_step(self, batch_data_dict):
        x = batch_data_dict['wav']
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)
        y_hat = self(x)
        return y_hat


class EventInvariantAdversarialTraining(LitAcousticClassificationSystem):
    def __init__(self,
                 optimizer_cfg: Dict,
                 scheduler_cfg: Optional[Dict] = None,
                 p_adv=0.5,
                 lr_scaling_factor=0.1,
                 adv_steps=1,
                 lambda_adv=1.0,
                 **kwargs):
        super(EventInvariantAdversarialTraining, self).__init__(**kwargs)
        self.p_adv = p_adv
        self.lambda_adv = lambda_adv
        self.lr_scaling_factor = lr_scaling_factor
        self.adv_steps = adv_steps
        self.optimizer_cfg = SimpleNamespace(**optimizer_cfg)
        self.scheduler_cfg = SimpleNamespace(**scheduler_cfg) if scheduler_cfg else None

        # Apply Mixup for both class_label and event_label
        self.mixup_label_keys = [self.class_label, "event"]

        # Build a new event classifier with a different number of classes
        self.event_classifier = build_new_classifier_from_old(self.backbone.classifier, new_num_classes=527)
        self.event_loss = torch.nn.BCEWithLogitsLoss()

        # Must set to False as we EIAT requires multiple optimizers
        self.automatic_optimization = False

    def configure_optimizers(self):
        # Build optimizers from YAML config
        opt_cls = hydra.utils.get_class(self.optimizer_cfg.class_path)
        opt_enc = opt_cls(self.backbone.parameters(), **self.optimizer_cfg.init_args)
        opt_event_cfg = self.optimizer_cfg.init_args
        # Scale the lr of event classifier
        opt_event_cfg['lr'] *= self.lr_scaling_factor
        opt_event = opt_cls(self.event_classifier.parameters(), **opt_event_cfg)
        # Case 1: Custom schedule
        if self.scheduler_cfg:
            sch_cls = hydra.utils.get_class(self.scheduler_cfg.class_path)
            sche_enc = sch_cls(opt_enc, **self.scheduler_cfg.init_args)
            sche_event = sch_cls(opt_event, **self.scheduler_cfg.init_args)
            return [opt_enc, opt_event], [sche_enc, sche_event]

        # Case 2: No scheduler
        return [opt_enc, opt_event], []

    @staticmethod
    def random_crop_batch_for_adv(tensor, prob: float):
        """
        Randomly crop a batch of shape (B, N) to (int(B*prob), N).
        """
        B = tensor.shape[0]
        k = int(B * prob)  # number of rows to keep
        indices = torch.randperm(B)[:k]  # random unique row indices
        return indices

    def training_step(self, batch_data_dict, batch_idx):
        if 'event' not in batch_data_dict:
            raise KeyError("Event labels are required for EIAT. Please enter 'soft_event_label_dir' in DCASEDataModule.")

        opt_enc, opt_event = self.optimizers()
        # Pre-processing: extract audio spectrogram and apply data augmentation
        x, labels = self.preprocess_batch(batch_data_dict)
        # Get the ground trues and predictions
        y = labels[self.class_label]
        e = labels['event']
        # Randomly select samples for adversarial training
        sample_idx_for_adv = self.random_crop_batch_for_adv(x, self.p_adv).to(self.device)

        # Step1: acoustic scene classification and event adversarial training
        # Compute alpha for GRL
        p_alpha = float(batch_idx + self.current_epoch * self.trainer.num_training_batches) \
                  / self.trainer.max_epochs / self.trainer.num_training_batches
        alpha_adv = 2. / (1. + np.exp(-10 * p_alpha)) - 1
        self.log_dict({'alpha': alpha_adv}, on_step=False, on_epoch=True, logger=True)
        # Apply extra steps to optimize the encoder with adversarial event loss
        for _ in range(self.adv_steps):
            # Get intermediate features
            y_hat, emb = self(x, return_emb=True)
            # Calculate the loss and accuracy
            train_scene_loss = self.compute_training_metrics(y_hat, y)
            # Apply gradient reversal layer (GRL)
            emb_adv = ReverseLayerF.apply(emb[sample_idx_for_adv], alpha_adv)
            e_hat = self.event_classifier(emb_adv)
            if hasattr(self.data_aug, 'mix_up'):
                adv_loss_event = self.data_aug.mix_up.cal_loss(e_hat, (e[0][sample_idx_for_adv], e[1][sample_idx_for_adv]), self.event_loss)
            else:
                adv_loss_event = self.event_loss(e_hat, e[sample_idx_for_adv])
            # Backprop
            opt_enc.zero_grad()
            self.manual_backward(train_scene_loss + self.lambda_adv * adv_loss_event)
            opt_enc.step()

        # Step2: event tagging
        # Get intermediate features
        _, emb = self(x, return_emb=True)
        # Calculate event loss
        e_hat = self.event_classifier(emb[sample_idx_for_adv])
        if hasattr(self.data_aug, 'mix_up'):
            train_loss_event = self.data_aug.mix_up.cal_loss(e_hat, (e[0][sample_idx_for_adv], e[1][sample_idx_for_adv]), self.event_loss)
        else:
            train_loss_event = self.event_loss(e_hat, e[sample_idx_for_adv])
        self.log_dict({f'train_loss_event': train_loss_event}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Backprop
        opt_event.zero_grad()
        self.manual_backward(self.lambda_adv * alpha_adv * train_loss_event)
        opt_event.step()

    def on_train_epoch_end(self):
        if self.lr_schedulers():
            sch_enc, sch_event = self.lr_schedulers()
            sch_enc.step()
            sch_event.step()

    def validation_step(self, batch_data_dict):
        x, y, e = batch_data_dict['wav'], batch_data_dict[self.class_label], batch_data_dict['event']
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)
        y_hat, emb = self(x, return_emb=True)
        val_loss_class = F.cross_entropy(y_hat, y)
        val_acc_class = self.accuracy(y_hat, y)
        self.log_dict({f'val_loss_{self.class_label}': val_loss_class, f'val_acc_{self.class_label}': val_acc_class},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)

        e_hat = self.event_classifier(emb)
        val_loss_event = self.event_loss(e_hat, e)
        self.log_dict({f'val_loss_event': val_loss_event}, on_step=False, on_epoch=True, prog_bar=False, logger=True)



class LitAscWithWarmupLinearDownScheduler(LitAcousticClassificationSystem):
    """
    ASC system with warmup-linear-down scheduler.
    """
    def __init__(self, optimizer: OptimizerCallable, warmup_len=4, down_len=26, min_lr=0.005, **kwargs):
        super(LitAscWithWarmupLinearDownScheduler, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.warmup_len = warmup_len
        self.down_len = down_len
        self.min_lr = min_lr

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        schedule_lambda = exp_warmup_linear_down(self.warmup_len, self.down_len, self.warmup_len, self.min_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

