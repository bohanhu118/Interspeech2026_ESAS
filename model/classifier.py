from torch import nn
import inspect


class _BaseClassifier(nn.Module):
    """ Base Module for classifiers. """


class ConvClassifier(_BaseClassifier):
    def __init__(self, in_channels: int, num_classes: int):
        super(ConvClassifier, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv = nn.Conv2d(in_channels, num_classes, 1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean((-1, -2), keepdim=False)
        return x


class SingleLinearClassifier(_BaseClassifier):
    def __init__(self, in_features: int, num_classes: int):
        super(SingleLinearClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = x.mean(dim=1) if x.dim() > 2 else x
        x = self.linear(x)
        return x


class MultiLayerPerception(_BaseClassifier):
    def __init__(self, in_features: int, hidden_units: int, num_classes: int, dropout: float):
        super(MultiLayerPerception, self).__init__()
        self.in_features = in_features
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.dropout = dropout

        self.fc1 = nn.Linear(in_features, hidden_units)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        x = x.mean((-1, -2), keepdim=False)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class ConvBnClassifier(_BaseClassifier):
    def __init__(self, in_channels: int, num_classes: int):
        super(ConvBnClassifier, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv = nn.Conv2d(in_channels, num_classes, 1, bias=False)
        self.bn = nn.BatchNorm2d(num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = x.mean((-1, -2), keepdim=False)
        return x

class LayerNormClassifier(_BaseClassifier):
    def __init__(self, in_features: int, num_classes: int):
        super(LayerNormClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.ln = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.ln(x)
        x = self.linear(x)
        return x


class Conv1dClassifier(_BaseClassifier):
    def __init__(self, in_channels: int, num_classes: int, kernel_size: int):
        super(Conv1dClassifier, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(in_channels, num_classes, kernel_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        logits = self.conv(x)
        logits = logits.view(logits.size(0), -1)
        return logits


import torch
from torchvision.ops.misc import Conv2dNormActivation
import torch.nn.functional as F
class GruClassifier(_BaseClassifier):
    def __init__(self, in_channels: int, mid_channels: int, num_classes: int):
        super(GruClassifier, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_classes = num_classes

        self.conv2d_4_1 = Conv2dNormActivation(
            in_channels=in_channels,
            out_channels=mid_channels,
            norm_layer=nn.BatchNorm2d,
            activation_layer=None,  # torch.nn.ReLU,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
            inplace=False)

        self.bgru2d_4_1 = nn.GRU(16, 64, num_layers=1, batch_first=True, bidirectional=False)
        self.drop1d_0_5 = nn.Dropout1d(p=0.5)
        self.param3 = nn.Parameter(torch.tensor(0.5))

        self.conv = nn.Conv1d(1, num_classes, mid_channels)

    def forward(self, x):
        x = F.silu(self.conv2d_4_1(x))
        x1 = torch.mean(x, dim=3)

        x, _ = self.bgru2d_4_1(x1)
        x2 = (self.drop1d_0_5(x))

        x1 = torch.mean(x1, dim=2)
        x2 = torch.mean(x2, dim=2)

        x = F.silu(self.param3 * x1 + (1 - self.param3) * x2)  # better than concat

        x = x.unsqueeze(1)
        logits = self.conv(x)
        logits = logits.view(logits.size(0), -1)
        return logits


def build_new_classifier_from_old(old_classifier, new_num_classes):
    # Get the keys of arguments of the original classifier and create a new dict
    init_params = inspect.signature(old_classifier.__class__.__init__).parameters
    new_params = {k: v.default for k, v in init_params.items() if k != 'self'}
    # Assign original values to new dict
    for k in new_params.keys():
        if hasattr(old_classifier, k):
            new_params[k] = getattr(old_classifier, k)
    new_params['num_classes'] = new_num_classes  # Change ``num_classes`` to scene number
    new_classifier = type(old_classifier)(
        **{key: new_params[key] for key in new_params if key in old_classifier.__init__.__code__.co_varnames})
    return new_classifier