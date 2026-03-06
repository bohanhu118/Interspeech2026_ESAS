import torch.nn as nn
from abc import ABC
from model.cpmobile_util import *
from model.beats.BEATs import BEATsConfig, BEATs
from model.classifier import ConvBnClassifier, Conv1dClassifier, GruClassifier, ConvClassifier, SingleLinearClassifier
from model.grucnn_uitl import get_ntu_model
from model.shared import ConvBnRelu, ResNorm, AdaResNorm, BroadcastBlock, TimeFreqSepConvolutions


class _BaseBackbone(nn.Module, ABC):
    """ Base Module for backbones. """
    def __init__(self,):
        super().__init__()
        self.classifier = None


class CPMobile(_BaseBackbone):
    def __init__(self, n_classes=10, in_channels=1, base_channels=32, channels_multiplier=2.3, expansion_rate=3.0,
              n_blocks=(3, 2, 1), strides=None):
        super(CPMobile, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channels_multiplier = channels_multiplier
        self.expansion_rate = expansion_rate
        self.n_blocks = n_blocks
        self.strides = strides if strides is not None else dict(b2=(2, 2), b4=(2, 1))
        self.n_stages = len(n_blocks)

        base_channels = make_divisible(base_channels, 8)
        channels_per_stage = [base_channels] + [make_divisible(base_channels * channels_multiplier ** stage_id, 8)
                                                for stage_id in range(self.n_stages)]
        self.total_block_count = 0

        self.in_c = nn.Sequential(
            ConvNormActivation(in_channels,
                               channels_per_stage[0] // 4,
                               kernel_size=3,
                               stride=2,
                               inplace=False
                               ),
            ConvNormActivation(channels_per_stage[0] // 4,
                               channels_per_stage[0],
                               activation_layer=torch.nn.ReLU,
                               kernel_size=3,
                               stride=2,
                               inplace=False
                               ),
        )

        self.stages = nn.Sequential()
        for stage_id in range(self.n_stages):
            stage = self._make_stage(channels_per_stage[stage_id],
                                     channels_per_stage[stage_id + 1],
                                     n_blocks[stage_id],
                                     strides=self.strides,
                                     expansion_rate=expansion_rate
                                     )
            self.stages.add_module(f"s{stage_id + 1}", stage)

        # ff_list = []
        # ff_list += [nn.Conv2d(
        #     channels_per_stage[-1],
        #     n_classes,
        #     kernel_size=(1, 1),
        #     stride=(1, 1),
        #     padding=0,
        #     bias=False),
        #     nn.BatchNorm2d(n_classes),
        # ]
        #
        # ff_list.append(nn.AdaptiveAvgPool2d((1, 1)))
        #
        # self.feed_forward = nn.Sequential(
        #     *ff_list
        # )

        self.classifier = ConvBnClassifier(channels_per_stage[-1], n_classes)

        self.apply(initialize_weights)

    def _make_stage(self,
                    in_channels,
                    out_channels,
                    n_blocks,
                    strides,
                    expansion_rate):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_id = self.total_block_count + 1
            bname = f'b{block_id}'
            self.total_block_count = self.total_block_count + 1
            if bname in strides:
                stride = strides[bname]
            else:
                stride = (1, 1)

            block = self._make_block(
                in_channels,
                out_channels,
                stride=stride,
                expansion_rate=expansion_rate
            )
            stage.add_module(bname, block)

            in_channels = out_channels
        return stage

    def _make_block(self,
                    in_channels,
                    out_channels,
                    stride,
                    expansion_rate,
                    ):

        block = CPMobileBlock(in_channels,
                              out_channels,
                              expansion_rate,
                              stride
                              )
        return block

    def _forward_conv(self, x):
        x = self.in_c(x)
        x = self.stages(x)
        return x

    def forward(self, x):
        global first_RUN
        emb = self._forward_conv(x)
        logits = self.classifier(emb)
        return logits, emb


class GRUCnn(_BaseBackbone):
    def __init__(self, n_classes=10, in_channels=1, n_blocks=(3, 2, 1), base_channels=16, channels_multiplier=1.5,
                 expansion_rate=2, divisor=3, strides=None):
        super(GRUCnn, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channels_multiplier = channels_multiplier
        self.expansion_rate = expansion_rate
        self.divisor = divisor
        self.n_blocks = n_blocks
        self.strides = strides

        self.encoder = get_ntu_model(n_classes, in_channels, n_blocks, base_channels, channels_multiplier,
                                     expansion_rate, divisor, strides)
        # self.classifier = Conv1dClassifier(1, n_classes, int(base_channels * channels_multiplier * expansion_rate * expansion_rate / divisor))
        in_channels_cla = int(base_channels * channels_multiplier * expansion_rate * expansion_rate)
        mid_channels_cla = int(base_channels * channels_multiplier * expansion_rate * expansion_rate / divisor)
        self.classifier = GruClassifier(in_channels_cla, mid_channels_cla, n_classes)

    def forward(self, x):
        emb = self.encoder(x)
        logits = self.classifier(emb)
        return logits, emb


class TFSepNet(_BaseBackbone):
    """
    Implementation of TF-SepNet-64, based on Time-Frequency Separate Convolutions. Check more details at:
    https://ieeexplore.ieee.org/abstract/document/10447999 and
    https://dcase.community/documents/challenge2024/technical_reports/DCASE2024_Cai_61_t1.pdf

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        base_channels (int): Number of base channels that controls the complexity of model.
        depth (int): Network depth with two options: 16 or 17. When depth = 17, an additional Max-pooling layer is inserted before the last TF-SepConvs black.
        kernel_size (int): Kernel size of each convolutional layer in TF-SepConvs blocks.
        dropout (float): Dropout rate.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, base_channels: int = 64, depth: int = 17,
                 kernel_size: int = 3, dropout: float = 0.1):
        super(TFSepNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.depth = depth
        assert base_channels % 2 == 0, "Base_channels should be divisible by 2."
        self.dropout = dropout
        self.kernel_size = kernel_size

        # Two settings of the depth. ``17`` have an additional Max-pooling layer before the final block of TF-SepConvs.
        cfg = {
            16: ['N', 1, 1, 'N', 'M', 1.5, 1.5, 'N', 'M', 2, 2, 'N', 2.5, 2.5, 2.5, 'N'],
            17: ['N', 1, 1, 'N', 'M', 1.5, 1.5, 'N', 'M', 2, 2, 'N', 'M', 2.5, 2.5, 2.5, 'N'],
        }

        self.conv_layers = nn.Sequential(ConvBnRelu(in_channels, base_channels // 2, 3, stride=2, padding=1),
                                         ConvBnRelu(base_channels // 2, 2 * base_channels, 3, stride=2, padding=1,
                                                    groups=base_channels // 2))
        # Compute the number of channels for each layer.
        layer_config = [int(i * base_channels) if not isinstance(i, str) else i for i in cfg[depth]]
        self.middle_layers = self._make_layers(base_channels, layer_config)
        # Get the index of channel number for the cla_layer.
        last_num_index = -1 if not isinstance(layer_config[-1], str) else -2
        # 1x1 convolution layer as the cla_layer.
        self.classifier = ConvClassifier(layer_config[last_num_index], num_classes)

    def _make_layers(self, width: int, layer_config: list):
        layers = []
        vt = width * 2
        for v in layer_config:
            if v == 'N':
                layers += [ResNorm(channels=vt)]
            elif v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v != vt:
                layers += [TimeFreqSepConvolutions(vt, v, self.kernel_size, self.dropout)]
                vt = v
            else:
                layers += [TimeFreqSepConvolutions(vt, vt, self.kernel_size, self.dropout)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        emb = self.middle_layers(x)
        return self.classifier(emb), emb


class BCResNet(_BaseBackbone):
    """
    Implementation of BC-ResNet, based on Broadcasted Residual Learning.
    Check more details at: https://arxiv.org/abs/2106.04140

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        base_channels (int): Number of base channels that controls the complexity of model.
        depth (int): Network depth with single option: 15.
        kernel_size (int): Kernel size of each convolutional layer in BC blocks.
        dropout (float): Dropout rate.
        sub_bands (int): Number of sub-bands for SubSpectralNorm (SSN). ``1`` indicates SSN is not applied.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, base_channels: int = 40, depth: int = 15,
                 kernel_size: int = 3, dropout: float = 0.1, sub_bands: int = 1):
        super(BCResNet, self).__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.sub_bands = sub_bands

        cfg = {
            15: ['N', 1, 1, 'N', 'M', 1.5, 1.5, 'N', 'M', 2, 2, 'N', 2.5, 2.5, 2.5, 'N'],
        }

        self.conv_layers = nn.Conv2d(in_channels, 2 * base_channels, 5, stride=2, bias=False, padding=2)
        # Compute the number of channels for each layer.
        layer_config = [int(i * base_channels) if not isinstance(i, str) else i for i in cfg[depth]]
        self.middle_layers = self._make_layers(base_channels, layer_config)
        # Get the index of channel number for the cla_layer.
        last_num_index = -1 if not isinstance(layer_config[-1], str) else -2
        # 1x1 convolution layer as the cla_layer.
        self.classifier = ConvClassifier(layer_config[last_num_index], num_classes)

    def _make_layers(self, width: int, layer_config: list):
        layers = []
        vt = 2 * width
        for v in layer_config:
            if v == 'N':
                layers += [ResNorm(channels=vt)]
            elif v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v != vt:
                layers += [BroadcastBlock(vt, v, self.kernel_size, self.dropout, self.sub_bands)]
                vt = v
            else:
                layers += [BroadcastBlock(vt, vt, self.kernel_size, self.dropout, self.sub_bands)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.middle_layers(x)
        # Get embedding features before classification layer
        emb = x
        x = self.classifier(x)
        return x, emb


class PretrainedBEATs(_BaseBackbone):
    """
    Module wrapping a BEATs encoder with pretrained weights and a new linear classifier.
    Check more details at: https://arxiv.org/abs/2212.09058
    """

    def __init__(self, pretrained=None, num_classes=10, **kwargs):
        super(PretrainedBEATs, self).__init__()

        if pretrained:
            try:
                ckpt = torch.load(pretrained, map_location='cpu', weights_only=False)
            except TypeError:
                ckpt = torch.load(pretrained, map_location='cpu')
            hyperparams = ckpt['cfg']
        else:
            ckpt = None
            hyperparams = kwargs

        cfg = BEATsConfig(hyperparams)
        self.encoder = BEATs(cfg)
        if pretrained:
            self.encoder.load_state_dict(ckpt['model'], strict=False)

        self.classifier = SingleLinearClassifier(in_features=cfg.encoder_embed_dim, num_classes=num_classes)

        # Move encoder to eval mode for faster inference
        self.encoder.eval()

    def forward(self, x):
        """
        Optimized forward pass that keeps data on the same device.
        """
        # Remove channel dimension and prepare for patch embedding
        x = x.squeeze(1).unsqueeze(1)  # [batch, 1, time, mel]

        # Patch embedding
        features = self.encoder.patch_embedding(x)

        # Reshape: [batch, embed_dim, time/patch, mel/patch] -> [batch, num_patches, embed_dim]
        features = features.flatten(2).transpose(1, 2)

        # Layer norm and post processing
        features = self.encoder.layer_norm(features)
        if self.encoder.post_extract_proj is not None:
            features = self.encoder.post_extract_proj(features)
        features = self.encoder.dropout_input(features)

        # Transformer encoder
        features, _ = self.encoder.encoder(features, padding_mask=None)

        # Global pooling and classification
        embeddings = features.mean(dim=1)
        logits = self.classifier(embeddings)

        return logits, embeddings