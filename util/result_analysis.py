import os
import numpy as np
import lightning.pytorch as pl
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE
from typing import Dict, Any, Tuple, Optional, Union, List
import lightning as L
import torch
import pandas as pd
import seaborn as sns
import io
from PIL import Image
from collections import defaultdict


def make_markdown_table(array):
    """ Convert the array-like classification report into a markdown table """

    nl = "\n"

    markdown = nl
    markdown += f"| {' | '.join(array[0])} |"

    markdown += nl
    markdown += f"| {' | '.join(['---'] * len(array[0]))} |"

    markdown += nl
    for entry in array[1:]:
        markdown += f"| {' | '.join(entry)} |{nl}"
    return markdown


class ClassificationSummaryCallback(pl.callbacks.Callback):
    """
    Analyze the classification results with a class-domain table report and a confusion matrix.
    """
    def __init__(self, feature_to_visualize=None):
        visual_features = {'embedding': "emb", 'logit': "y_hat"}
        self.feature_to_visualize = visual_features[feature_to_visualize] if feature_to_visualize is not None else None

        self.class_labels = None
        self.domain_labels = None

        # Save data during testing for statistical analysis
        self._test_step_outputs = {'emb': [], 'y': [], 'pred': [], 'd': [], 'y_hat': []}

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        try:
            self.class_labels = trainer.datamodule.test_set.scene_classes
            if hasattr(trainer.datamodule.test_set, 'domain_classes'):
                self.domain_labels = trainer.datamodule.test_set.domain_classes
        except Exception as e:
            # Raise immediately when an invalid file is found
            raise RuntimeError(f"The test_set in data_module is not properly implemented. \nError: {e}")

    def get_table_report(self, inputs):
        _y_true = inputs['y']
        _y_pred = inputs['pred']
        _d_indices = inputs['d']
        # Convert device indices to device labels
        d = [self.domain_labels[i] for i in _d_indices]
        # Create a dictionary to store the class-wise accuracy for each domain
        domain_class_accuracy = {}
        for domain_label in self.domain_labels:
            indices = [i for i, x in enumerate(d) if x == domain_label]
            domain_true = [_y_true[i] for i in indices]
            domain_pred = [_y_pred[i] for i in indices]
            class_accuracy = {}
            for class_index in set(domain_true):
                indices = [i for i, x in enumerate(domain_true) if x == class_index]
                class_true = [domain_true[i] for i in indices]
                class_pred = [domain_pred[i] for i in indices]
                class_accuracy[class_index] = accuracy_score(class_true, class_pred)
            domain_class_accuracy[domain_label] = class_accuracy
        # Create a table-like output of domain-wise and class-wise accuracy
        column_names = ["Class"] + self.domain_labels + ["Class Avg."]
        classification_report = [column_names]
        for class_label in self.class_labels:
            row = [str(class_label)]
            class_index = self.class_labels.index(class_label)
            class_avg_acc = 0.0
            for domain_label in self.domain_labels:
                if class_index in domain_class_accuracy[domain_label]:
                    row.append(f"{domain_class_accuracy[domain_label][class_index] * 100:.2f}")
                    domain_weighted_accuracy = domain_class_accuracy[domain_label][class_index] * d.count(domain_label) / len(d)
                    class_avg_acc += domain_weighted_accuracy
                else:
                    row.append("N/A")
            classification_report.append(row + [f"{class_avg_acc * 100:.2f}"])
        # Add a row that shows the macro average accuracy across all domains and class_labels
        num_classes = len(self.class_labels)
        total_accuracy = 0.0
        domain_avg_row = ['Domain Avg.']
        for domain_label in self.domain_labels:
            domain_accuracy = 0.0
            domain_class_count = 0
            for class_label in range(num_classes):
                if class_label in domain_class_accuracy[domain_label]:
                    domain_weighted_accuracy = domain_class_accuracy[domain_label][class_label] * _y_true.count(class_label) / len(_y_true)
                    domain_accuracy += domain_weighted_accuracy
                    domain_class_count += 1
            if domain_class_count > 0:
                domain_weighted_accuracy = domain_accuracy * d.count(domain_label) / len(d)
                total_accuracy += domain_weighted_accuracy
            else:
                domain_accuracy = "N/A"
            domain_avg_row.append(f"{domain_accuracy * 100:.2f}")
        classification_report.append(domain_avg_row + [f"{total_accuracy * 100:.2f}"])
        markdown = make_markdown_table(classification_report)
        return markdown

    def get_confusion_matrix(self, inputs, save_to=None):
        _y_true = inputs['y']
        _y_pred = inputs['pred']
        # Compute confusion matrix
        cm = confusion_matrix(_y_true, _y_pred)
        if save_to:
            cm_path = os.path.join(save_to, 'confusion_matrix.npy')
            np.save(cm_path, cm)
        # Convert to probability confusion matrix
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(8, 8))
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]))
        ax.set_xticklabels(self.class_labels, fontsize=12)
        ax.set_yticklabels(self.class_labels, fontsize=12)
        ax.set_ylabel('True label', fontsize=14)
        ax.set_xlabel('Predicted label', fontsize=14)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        if save_to:
            fig_path = os.path.join(save_to, 'confusion_matrix.png')
            plt.savefig(fig_path, dpi=300)
        return fig

    def tsne_to_figure(self, features, label_indices, label_type, save_to=None):
        # Convert indices to labels
        if label_type == "class":
            label = self.class_labels
        elif label_type == "domain":
            label = self.domain_labels
        else:
            assert label_type in ["class", "domain"], "Label should be `class` or `domain`"
            return
        labels = [label[i] for i in label_indices]
        # Plot the t-SNE results
        fig, ax = plt.subplots(figsize=(8, 5))
        # Create a scatter plot with series
        for label in np.unique(labels):
            indices = []
            for i, x in enumerate(labels):
                if x == label:
                    indices.append(i)
            ax.scatter(features[indices, 0], features[indices, 1], label=label, s=6)
        # Create a legend
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        ax.set_aspect('equal', adjustable='box')
        # Hide the tick marks on x-axis and y-axis
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.subplots_adjust(right=0.625)
        if save_to:
            fig_path = os.path.join(save_to, f"tsne_visualize_{label_type}.png")
            plt.savefig(fig_path, dpi=300)
        return fig

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx, dataloader_idx=0):
        if self.class_labels is None:
            return
        if self.domain_labels:
            for k in ['y', 'y_hat', 'pred', 'd', 'emb']:
                self._test_step_outputs[k].extend(outputs[k].detach().cpu().numpy())
        else:
            for k in ['y', 'y_hat', 'pred', 'emb']:
                self._test_step_outputs[k].extend(outputs[k].detach().cpu().numpy())

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.class_labels is None:
            return
        tensorboard = pl_module.logger
        if self.domain_labels:
            # Generate a classification report table
            tab_report = self.get_table_report(self._test_step_outputs)
            tensorboard.experiment.add_text('classification_report', tab_report)
        # Generate an confusion matrix figure
        cm = self.get_confusion_matrix(self._test_step_outputs, save_to=tensorboard.log_dir)
        tensorboard.experiment.add_figure('confusion_matrix', cm)
        if self.feature_to_visualize:
            features = self._test_step_outputs[self.feature_to_visualize]
            # Convert list to numpy
            features = np.array(features)
            # Flatten the features
            features = np.reshape(features, newshape=(features.shape[0], -1))
            # Apply t-SNE to reduce dimensionality
            tsne = TSNE()
            tsne_results = tsne.fit_transform(features)
            # Generate a tsne visualization about the features across classes
            tsne_classes = self.tsne_to_figure(tsne_results, self._test_step_outputs['y'], label_type="class", save_to=tensorboard.log_dir)
            tensorboard.experiment.add_figure('tsne_visualize_class', tsne_classes)
            # Generate a tsne visualization about the features across domains
            tsne_domains = self.tsne_to_figure(tsne_results, self._test_step_outputs['d'], label_type="domain", save_to=tensorboard.log_dir)
            tensorboard.experiment.add_figure('tsne_visualize_domain', tsne_domains)
