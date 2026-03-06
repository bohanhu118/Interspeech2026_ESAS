import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as L
from typing import Dict, Any, Tuple, Optional, Union, List
from types import SimpleNamespace
from collections import defaultdict


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer for adversarial domain adaptation.
    Reverses gradient during backward pass with scaling factor alpha.

    Reference: Ganin & Lempitsky (2015), "Unsupervised Domain Adaptation by Backpropagation"
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.lambd = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha * self.lambd)

    def update_lambda(self, p: float):
        """
        Update lambda based on training progress.

        Args:
            p: Training progress in [0, 1]
        """
        # Schedule from Ganin et al. (2015)
        gamma = 10
        self.lambd = 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0


class GradientReversalFunction(torch.autograd.Function):
    """Custom autograd function for gradient reversal."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return ctx.alpha * -grad_output, None


class DomainClassifier(nn.Module):
    """
    Domain classifier for adversarial training.
    Classifies whether features come from source or target domain.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_domains: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains

        # Build network dynamically based on input dimension
        self._build_network()

        # Initialize weights properly
        self._initialize_weights()

    def _build_network(self):
        """Build network architecture based on input dimension."""
        layers = []

        # First layer
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(0.5))

        # Second layer (optional, depends on feature dimension)
        if self.input_dim >= 128:  # Only add if we have enough features
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim // 2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.5))
            output_layer_input = self.hidden_dim // 2
        else:
            output_layer_input = self.hidden_dim

        # Output layer
        layers.append(nn.Linear(output_layer_input, self.num_domains))

        self.network = nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights for better convergence."""
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pooling if input is 4D (B, C, H, W)
        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.flatten(1)
        elif x.dim() == 3:
            # Temporal features - average over time
            x = x.mean(dim=1)

        return self.network(x)


class LitEventRobustAdversarialTraining(L.LightningModule):
    """
    Progressive Event-Robust and Domain Adversarial Training for ASC.
    Gradually introduces adversarial losses to prevent negative transfer.

    Args:
        backbone: Scene encoder backbone
        data_augmentation: Data augmentation techniques
        class_label: Scene label key ('scene')
        spec_extractor: Spectrogram extractor
        lambda_domain: Final weight for domain adversarial loss
        lambda_event: Final weight for event adversarial loss
        adaptive_alpha: Whether to use adaptive GRL scheduling
        grl_alpha: Initial GRL scaling factor
        domain_hidden_dim: Hidden dimension for domain classifier
        event_tag_dim: Dimension of event tags (default: 527)
        num_domains: Number of domain classes
        warmup_len: Number of warmup epochs
        down_len: Number of decay epochs
        min_lr: Minimum learning rate
        optimizer: Optimizer callable
        mix_type_as_domain: Whether to use mix_type for multi-class domain adaptation
        feature_layer: Name of the layer to extract features from backbone
        feature_normalize: Whether to normalize features before classifiers
        unsupervised_da: If True, target samples have NO scene labels
        separate_source_target_aug: If True, use separate augmentation for source and target
        # Progressive training parameters
        domain_warmup_epochs: Epochs before starting domain adaptation (default: 30)
        lambda_domain_schedule: Whether to schedule lambda_domain (default: True)
        lambda_domain_start: Starting value for lambda_domain (default: 0.1)
        lambda_domain_ramp_epochs: Epochs to ramp up lambda_domain (default: 50)
        event_warmup_epochs: Epochs before starting event adaptation (default: 20)
        lambda_event_schedule: Whether to schedule lambda_event (default: True)
        lambda_event_start: Starting value for lambda_event (default: 0.0)
        lambda_event_ramp_epochs: Epochs to ramp up lambda_event (default: 80)
        consistency_weight: Weight for feature consistency loss (default: 0.3)
    """

    def __init__(self,
                 backbone: nn.Module,
                 data_augmentation: Dict[str, Any] = None,
                 class_label: str = 'scene',
                 spec_extractor: Any = None,
                 lambda_domain: float = 1.0,
                 lambda_event: float = 1.0,
                 adaptive_alpha: bool = True,
                 grl_alpha: float = 1.0,
                 domain_hidden_dim: int = 256,
                 event_tag_dim: int = 527,
                 num_domains: int = 2,
                 warmup_len: int = 14,
                 down_len: int = 84,
                 min_lr: float = 0.005,
                 optimizer: Any = None,
                 mix_type_as_domain: bool = False,
                 feature_layer: Optional[str] = None,
                 feature_normalize: bool = True,
                 unsupervised_da: bool = True,
                 separate_source_target_aug: bool = True,
                 # Progressive training parameters
                 domain_warmup_epochs: int = 30,
                 lambda_domain_schedule: bool = True,
                 lambda_domain_start: float = 0.1,
                 lambda_domain_ramp_epochs: int = 50,
                 event_warmup_epochs: int = 20,
                 lambda_event_schedule: bool = True,
                 lambda_event_start: float = 0.0,
                 lambda_event_ramp_epochs: int = 80,
                 consistency_weight: float = 0.3,
                 **kwargs):

        super().__init__()

        # Store base components
        self.backbone = backbone
        self.class_label = class_label
        self.spec_extractor = spec_extractor
        self.mix_type_as_domain = mix_type_as_domain
        self.feature_layer = feature_layer
        self.feature_normalize = feature_normalize
        self.unsupervised_da = unsupervised_da
        self.separate_source_target_aug = separate_source_target_aug

        # Store progressive training parameters
        self.domain_warmup_epochs = domain_warmup_epochs
        self.lambda_domain_schedule = lambda_domain_schedule
        self.lambda_domain_start = lambda_domain_start
        self.lambda_domain_ramp_epochs = lambda_domain_ramp_epochs
        self.event_warmup_epochs = event_warmup_epochs
        self.lambda_event_schedule = lambda_event_schedule
        self.lambda_event_start = lambda_event_start
        self.lambda_event_ramp_epochs = lambda_event_ramp_epochs
        self.consistency_weight = consistency_weight

        # Store final lambda values for reference
        self.lambda_domain_final = lambda_domain
        self.lambda_event_final = lambda_event

        # Adversarial training parameters
        self.lambda_domain = lambda_domain
        self.lambda_event = lambda_event
        self.adaptive_alpha = adaptive_alpha
        self.grl_alpha = grl_alpha
        self.domain_hidden_dim = domain_hidden_dim
        self.event_tag_dim = event_tag_dim
        self.num_domains = num_domains

        # Learning rate scheduler parameters
        self.warmup_len = warmup_len
        self.down_len = down_len
        self.min_lr = min_lr
        self.optimizer = optimizer or torch.optim.Adam

        # Data augmentation
        self.data_aug = SimpleNamespace(**data_augmentation) if data_augmentation else SimpleNamespace()
        self.mixup_label_keys = [self.class_label, 'event']

        # Initialize classifiers (they will be initialized later with proper feature dim)
        self.domain_classifier = None
        self.event_classifier = None

        # GRL layers for different adversarial tasks
        self.domain_grl = GradientReversalLayer(alpha=grl_alpha)
        self.event_grl = GradientReversalLayer(alpha=grl_alpha)
        self.target_domain_grl = GradientReversalLayer(alpha=grl_alpha)

        # Training tracking
        self.current_progress = 0.0
        self._feature_dim = None
        self._classifiers_initialized = False

        # Track current epoch for progressive scheduling
        self.current_epoch_tracking = 0

        # Store feature dimension for later initialization
        self._estimated_feature_dim = None

        # GRL performance monitoring
        self.grl_metrics = defaultdict(list)
        self.epoch_grl_summary = {}

        self.save_hyperparameters(ignore=['backbone', 'spec_extractor', 'optimizer'])

    def _get_current_lambda_domain(self) -> float:
        """
        Get current lambda_domain value based on progressive schedule.

        Returns:
            Current lambda_domain value
        """
        if not self.lambda_domain_schedule:
            return self.lambda_domain_final

        current_epoch = self.current_epoch_tracking

        # Phase 1: Warmup - no domain adaptation
        if current_epoch < self.domain_warmup_epochs:
            return 0.0

        # Phase 2: Ramp up - gradually increase lambda_domain
        elif current_epoch < self.domain_warmup_epochs + self.lambda_domain_ramp_epochs:
            progress = (current_epoch - self.domain_warmup_epochs) / self.lambda_domain_ramp_epochs
            return self.lambda_domain_start + (self.lambda_domain_final - self.lambda_domain_start) * progress

        # Phase 3: Full strength
        else:
            return self.lambda_domain_final

    def _get_current_lambda_event(self) -> float:
        """
        Get current lambda_event value based on progressive schedule.

        Returns:
            Current lambda_event value
        """
        if not self.lambda_event_schedule:
            return self.lambda_event_final

        current_epoch = self.current_epoch_tracking

        # Phase 1: Warmup - no event adaptation
        if current_epoch < self.event_warmup_epochs:
            return 0.0

        # Phase 2: Ramp up - gradually increase lambda_event
        elif current_epoch < self.event_warmup_epochs + self.lambda_event_ramp_epochs:
            progress = (current_epoch - self.event_warmup_epochs) / self.lambda_event_ramp_epochs
            return self.lambda_event_start + (self.lambda_event_final - self.lambda_event_start) * progress

        # Phase 3: Full strength
        else:
            return self.lambda_event_final

    def _compute_feature_consistency_loss(self, features_source: torch.Tensor,
                                          features_target: torch.Tensor) -> torch.Tensor:
        """
        Compute feature consistency loss to prevent negative transfer.
        Encourages source and target features to have similar statistics.

        Args:
            features_source: Features from source domain
            features_target: Features from target domain

        Returns:
            Consistency loss
        """
        if features_source.size(0) == 0 or features_target.size(0) == 0:
            return torch.tensor(0.0, device=self.device)

        # Compute mean and covariance statistics
        source_mean = features_source.mean(dim=0)
        target_mean = features_target.mean(dim=0)

        # Mean alignment loss
        mean_loss = F.mse_loss(source_mean, target_mean)

        # Covariance alignment loss (simplified)
        source_cov = torch.mm(features_source.t(), features_source) / features_source.size(0)
        target_cov = torch.mm(features_target.t(), features_target) / features_target.size(0)
        cov_loss = F.mse_loss(source_cov, target_cov)

        # Combined consistency loss
        consistency_loss = mean_loss + cov_loss

        return consistency_loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step with progressive adversarial training."""
        if self.device.type == 'mps':
            torch.mps.empty_cache()

        if not (isinstance(batch, dict) and 'source' in batch and 'target' in batch):
            raise ValueError("Batch must contain 'source' and 'target' keys")

        source_batch = batch['source']
        target_batch = batch['target']

        # Update current epoch tracking
        self.current_epoch_tracking = self.current_epoch

        # Get current lambda values based on progressive schedule
        current_lambda_domain = self._get_current_lambda_domain()
        current_lambda_event = self._get_current_lambda_event()

        # Compute dynamic alpha for GRL
        alpha = self._compute_dynamic_alpha(batch_idx)

        # Prepare batch
        x_all, y_source, y_event, d_all, source_mask = self._prepare_clean_domain_adaptation_batch(
            source_batch, target_batch, batch_idx
        )

        # Create domain mask for GRL
        domain_mask = ~source_mask

        # Forward pass through backbone
        scene_logits, features = self._extract_backbone_features(x_all)

        if isinstance(scene_logits, tuple):
            scene_logits = scene_logits[0]

        # Prepare features for classifiers
        features = self._prepare_features_for_classifier(features)

        # Ensure classifiers are initialized (CRITICAL FIX)
        if not self._classifiers_initialized:
            feature_dim = features.shape[1]
            self._ensure_classifiers_initialized(feature_dim)
        elif self._feature_dim is None:
            # Store feature dimension if not already stored
            self._feature_dim = features.shape[1]

        # ===== Scene classification loss (ALWAYS active) =====
        scene_logits_source = scene_logits[source_mask]

        if isinstance(y_source, tuple):
            # MixUp case
            if len(y_source) == 2:
                y_a, y_b = y_source
                lam = self.data_aug.mix_up.lam if hasattr(self.data_aug, 'mix_up') else 0.5
                loss_a = F.cross_entropy(scene_logits_source, y_a)
                loss_b = F.cross_entropy(scene_logits_source, y_b)
                scene_loss = lam * loss_a + (1 - lam) * loss_b

                pred = torch.argmax(scene_logits_source, dim=1)
                acc_a = (pred == y_a).float().mean()
                acc_b = (pred == y_b).float().mean()
                scene_acc = (lam * acc_a + (1 - lam) * acc_b).item()
            else:
                raise ValueError(f"Unsupported MixUp format")
        else:
            # Normal case
            scene_loss = F.cross_entropy(scene_logits_source, y_source)
            pred = torch.argmax(scene_logits_source, dim=1)
            scene_acc = (pred == y_source).float().mean().item()

        # Initialize domain and event losses
        domain_loss = torch.tensor(0.0, device=self.device)
        domain_acc = 0.5  # Random chance baseline
        event_loss = torch.tensor(0.0, device=self.device)

        # ===== Domain classification with GRL (conditional) =====
        source_domain_acc = 0.5
        target_domain_acc = 0.5
        if current_lambda_domain > 0:
            # Apply GRL based on domain mask
            features_domain = features.clone()

            # Target domain features get reversed gradient
            self.target_domain_grl.alpha = alpha
            features_domain[domain_mask] = self.target_domain_grl(features[domain_mask])

            # Source domain features pass through without GRL
            features_domain[~domain_mask] = self.domain_grl(features[~domain_mask])

            domain_logits = self.domain_classifier(features_domain)

            # Compute domain loss
            domain_loss = F.cross_entropy(domain_logits, d_all)
            domain_preds = torch.argmax(domain_logits, dim=1)
            domain_acc = (domain_preds == d_all).float().mean().item()

            # Collect GRL performance metrics for epoch summary
            source_mask_bool = ~domain_mask  # Source samples (domain 0)
            target_mask_bool = domain_mask  # Target samples (domain 1)

            if source_mask_bool.any():
                source_domain_preds = domain_preds[source_mask_bool]
                source_domain_labels = d_all[source_mask_bool]
                source_domain_acc = (source_domain_preds == source_domain_labels).float().mean().item()
                self.grl_metrics['source_acc'].append(source_domain_acc)

            if target_mask_bool.any():
                target_domain_preds = domain_preds[target_mask_bool]
                target_domain_labels = d_all[target_mask_bool]
                target_domain_acc = (target_domain_preds == target_domain_labels).float().mean().item()
                self.grl_metrics['target_acc'].append(target_domain_acc)

            # Store other metrics
            self.grl_metrics['domain_acc'].append(domain_acc)
            self.grl_metrics['domain_loss'].append(domain_loss.item())
            self.grl_metrics['grl_alpha'].append(alpha)
            self.grl_metrics['lambda_domain'].append(current_lambda_domain)
            self.grl_metrics['lambda_event'].append(current_lambda_event)

            # Log domain-specific accuracies
            self._log_domain_specific_accuracies(domain_preds, d_all)
        else:
            # Domain adaptation not active yet
            domain_logits = None
            # Still collect metrics for reference
            self.grl_metrics['source_acc'].append(0.5)
            self.grl_metrics['target_acc'].append(0.5)
            self.grl_metrics['domain_acc'].append(0.5)
            self.grl_metrics['domain_loss'].append(0.0)
            self.grl_metrics['grl_alpha'].append(alpha)
            self.grl_metrics['lambda_domain'].append(current_lambda_domain)
            self.grl_metrics['lambda_event'].append(current_lambda_event)

        # ===== Event classification with GRL (conditional) =====
        event_logits = None
        if current_lambda_event > 0 and y_event is not None:
            # Apply event GRL to all features
            self.event_grl.alpha = alpha
            features_event = self.event_grl(features)
            event_logits = self.event_classifier(features_event)

            # Handle different formats of y_event
            if isinstance(y_event, tuple):
                # MixUp case for event tags
                if len(y_event) == 2:
                    y_event_a, y_event_b = y_event
                    # Get event logits for source domain only
                    event_logits_source = event_logits[source_mask]

                    # Compute binary cross-entropy for both MixUp components
                    loss_a = F.binary_cross_entropy_with_logits(event_logits_source, y_event_a)
                    loss_b = F.binary_cross_entropy_with_logits(event_logits_source, y_event_b)

                    # Use same lambda as scene MixUp
                    lam = self.data_aug.mix_up.lam if hasattr(self.data_aug, 'mix_up') else 0.5
                    event_loss = lam * loss_a + (1 - lam) * loss_b
            else:
                # Regular tensor case
                # Get event logits for source domain only
                event_logits_source = event_logits[source_mask]

                # Check if y_event includes target domain
                if y_event.size(0) == len(d_all):
                    # y_event includes both source and target, extract source only
                    y_event_source = y_event[source_mask]
                else:
                    # y_event is source only
                    y_event_source = y_event

                # Binary cross-entropy for multi-label classification
                event_loss = F.binary_cross_entropy_with_logits(
                    event_logits_source, y_event_source
                )

        # ===== Feature consistency loss (ALWAYS active) =====
        consistency_loss = torch.tensor(0.0, device=self.device)
        if self.consistency_weight > 0:
            features_source = features[source_mask]
            features_target = features[~source_mask]
            if features_source.size(0) > 0 and features_target.size(0) > 0:
                consistency_loss = self._compute_feature_consistency_loss(
                    features_source, features_target
                )

        # ===== Total loss with progressive weights =====
        total_loss = (scene_loss +
                      current_lambda_domain * domain_loss +
                      current_lambda_event * event_loss +
                      self.consistency_weight * consistency_loss)

        # ===== Logging =====
        self.log('train_scene_loss', scene_loss, prog_bar=True)
        self.log('train_scene_acc', scene_acc, prog_bar=True)

        if current_lambda_domain > 0:
            self.log('train_domain_loss', domain_loss, prog_bar=False)
            self.log('train_domain_acc', domain_acc, prog_bar=False)

        if current_lambda_event > 0 and y_event is not None:
            self.log('train_event_loss', event_loss, prog_bar=False)

        if self.consistency_weight > 0:
            self.log('train_consistency_loss', consistency_loss, prog_bar=False)

        self.log('train_total_loss', total_loss, prog_bar=True)
        self.log('grl_alpha', alpha, on_step=True, on_epoch=True, prog_bar=True)
        self.log('current_lambda_domain', current_lambda_domain, prog_bar=True)
        self.log('current_lambda_event', current_lambda_event, prog_bar=True)

        # Log training phase
        if self.current_epoch_tracking < self.domain_warmup_epochs:
            phase = "scene_only"
        elif self.current_epoch_tracking < self.domain_warmup_epochs + self.lambda_domain_ramp_epochs:
            phase = "domain_ramp"
        elif self.current_epoch_tracking < self.event_warmup_epochs:
            phase = "domain_full"
        elif self.current_epoch_tracking < self.event_warmup_epochs + self.lambda_event_ramp_epochs:
            phase = "event_ramp"
        else:
            phase = "full_training"

        self.log('train_phase', {'scene_only': 0, 'domain_ramp': 1, 'domain_full': 2,
                                 'event_ramp': 3, 'full_training': 4}[phase], prog_bar=False)

        source_ratio = source_mask.float().mean()
        target_ratio = (~source_mask).float().mean()
        return total_loss

    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        super().on_train_epoch_end()

        # Calculate average GRL metrics for the epoch
        if self.grl_metrics:
            epoch_summary = {}
            for key, values in self.grl_metrics.items():
                if values:  # Check if list is not empty
                    epoch_summary[f'avg_{key}'] = np.mean(values)

            # Store the summary for this epoch
            self.epoch_grl_summary[self.current_epoch] = epoch_summary

            # Print GRL performance summary
            self._print_grl_summary(epoch_summary)

            # Clear metrics for next epoch
            self.grl_metrics.clear()

    def _print_grl_summary(self, summary: Dict[str, float]):
        """Print formatted GRL performance summary."""
        epoch = self.current_epoch
        current_lambda_domain = self._get_current_lambda_domain()
        current_lambda_event = self._get_current_lambda_event()

        print(f"\n{'=' * 80}")
        print(f"[GRL Epoch Summary] Epoch {epoch}:")
        print(f"{'=' * 80}")

        # Get training phase
        if epoch < self.domain_warmup_epochs:
            phase = "Scene-only training"
        elif epoch < self.domain_warmup_epochs + self.lambda_domain_ramp_epochs:
            phase = "Domain adaptation ramping up"
        elif epoch < self.event_warmup_epochs:
            phase = "Full domain adaptation"
        elif epoch < self.event_warmup_epochs + self.lambda_event_ramp_epochs:
            phase = "Event adaptation ramping up"
        else:
            phase = "Full adversarial training"

        print(f"  - Training Phase: {phase}")
        print(f"  - Lambda_domain: {current_lambda_domain:.3f}")
        print(f"  - Lambda_event: {current_lambda_event:.3f}")
        print(f"  - GRL Alpha: {summary.get('avg_grl_alpha', 1.0):.3f}")

        # Domain classifier performance
        if 'avg_domain_acc' in summary:
            print(f"\n  Domain Classifier Performance:")
            print(f"  - Overall domain acc: {summary['avg_domain_acc']:.3f}")
            print(f"  - Source domain acc: {summary.get('avg_source_acc', 0.5):.3f}")
            print(f"  - Target domain acc: {summary.get('avg_target_acc', 0.5):.3f}")
            print(f"  - Domain loss: {summary.get('avg_domain_loss', 0.0):.4f}")

            # Performance analysis
            source_acc = summary.get('avg_source_acc', 0.5)
            target_acc = summary.get('avg_target_acc', 0.5)

            print(f"\n  GRL Effectiveness Analysis:")
            if source_acc < 0.6 and target_acc < 0.6:
                print(f"  ✓ GRL working well! Domain confusion is high (≈{min(source_acc, target_acc):.1%})")
            elif source_acc > 0.8 or target_acc > 0.8:
                print(
                    f"  ⚠️  GRL may be too weak! Domain separation is high (source:{source_acc:.1%}, target:{target_acc:.1%})")
            else:
                print(f"  ↻ GRL is learning... (source:{source_acc:.1%}, target:{target_acc:.1%})")

            # Check for balanced performance
            if abs(source_acc - target_acc) > 0.3:
                print(f"  ⚠️  Domain imbalance detected! Difference: {abs(source_acc - target_acc):.2f}")
        else:
            print(f"\n  Domain Classifier: Not active yet (warmup phase)")

        print(f"{'=' * 80}")

    def _ensure_classifiers_initialized(self, feature_dim: int):
        """Initialize domain and event classifiers."""
        current_device = self.device
        self._feature_dim = feature_dim

        # Domain classifier
        if self.domain_classifier is None:
            self.domain_classifier = DomainClassifier(
                input_dim=feature_dim,
                hidden_dim=self.domain_hidden_dim,
                num_domains=self.num_domains
            )
            self.domain_classifier.to(current_device)

        # Event classifier (527 classes for AudioSet)
        if self.event_classifier is None:
            self.event_classifier = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.event_tag_dim)
            )
            self.event_classifier.to(current_device)
        self._classifiers_initialized = True

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when loading a checkpoint."""
        super().on_load_checkpoint(checkpoint)

        # Try to extract feature dimension from checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']

            # Check for domain classifier weights
            domain_weight_key = None
            for key in state_dict.keys():
                if key.startswith('domain_classifier.network.0.weight'):
                    domain_weight_key = key
                    break

            if domain_weight_key:
                weight_shape = state_dict[domain_weight_key].shape
                if len(weight_shape) == 2:
                    feature_dim = weight_shape[1]
                    self._feature_dim = feature_dim
                    # Initialize classifiers immediately
                    self._ensure_classifiers_initialized(feature_dim)

    def on_fit_start(self):
        """Called at the beginning of training."""
        super().on_fit_start()

        # Ensure classifiers are initialized
        if not self._classifiers_initialized:
            try:
                # Try to infer feature dimension from backbone
                if self._estimated_feature_dim is not None:
                    feature_dim = self._estimated_feature_dim
                    self._ensure_classifiers_initialized(feature_dim)
                else:
                    # Create a dummy input to extract feature dimension
                    dummy_input = torch.randn(1, 1, 64, 100).to(self.device)
                    _, dummy_features = self._extract_backbone_features(dummy_input)
                    dummy_features = self._prepare_features_for_classifier(dummy_features)
                    feature_dim = dummy_features.shape[1]
                    self._ensure_classifiers_initialized(feature_dim)
            except Exception as e:
                print(f"[WARNING] Could not initialize classifiers: {e}")

        # Move all components to device
        self._move_to_device(self.device)

    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        super().on_train_epoch_start()

        # Log current training phase
        current_epoch = self.current_epoch
        if current_epoch < self.domain_warmup_epochs:
            print(f"[PHASE] Epoch {current_epoch}: Scene-only training (no domain/event adaptation)")
        elif current_epoch < self.domain_warmup_epochs + self.lambda_domain_ramp_epochs:
            progress = (current_epoch - self.domain_warmup_epochs) / self.lambda_domain_ramp_epochs
            lambda_domain = self._get_current_lambda_domain()
            print(f"[PHASE] Epoch {current_epoch}: Domain adaptation ramping up "
                  f"(lambda_domain={lambda_domain:.3f}, progress={progress:.2f})")
        elif current_epoch < self.event_warmup_epochs:
            print(f"[PHASE] Epoch {current_epoch}: Full domain adaptation, event adaptation pending")
        elif current_epoch < self.event_warmup_epochs + self.lambda_event_ramp_epochs:
            progress = (current_epoch - self.event_warmup_epochs) / self.lambda_event_ramp_epochs
            lambda_event = self._get_current_lambda_event()
            print(f"[PHASE] Epoch {current_epoch}: Event adaptation ramping up "
                  f"(lambda_event={lambda_event:.3f}, progress={progress:.2f})")
        else:
            print(f"[PHASE] Epoch {current_epoch}: Full adversarial training")

    def _prepare_clean_domain_adaptation_batch(self, source_batch, target_batch, batch_idx):
        """Prepare batch for training."""
        # Process source domain
        x_s = source_batch['wav']
        y_s_raw = source_batch[self.class_label]
        y_e_s = source_batch.get('event', None)
        d_s = torch.zeros(x_s.size(0), dtype=torch.long, device=self.device)

        # Apply data augmentation
        labels_s = {self.class_label: y_s_raw, 'domain': d_s}
        if y_e_s is not None:
            labels_s['event'] = y_e_s

        x_s, labels_s = self._apply_data_aug(x_s, labels_s, is_target_domain=False)
        y_s_processed = labels_s[self.class_label]
        y_e_s_processed = labels_s.get('event', y_e_s)

        # Process target domain
        x_t = target_batch['wav']
        d_t = torch.ones(x_t.size(0), dtype=torch.long, device=self.device)
        y_e_t = target_batch.get('event', None)

        # Handle target augmentation
        labels_t = {'domain': d_t}
        if y_e_t is not None:
            labels_t['event'] = y_e_t

        if not self.separate_source_target_aug or hasattr(self.data_aug, 'mix_up'):
            labels_t[self.class_label] = torch.full((x_t.size(0),), -1,
                                                    dtype=torch.long,
                                                    device=self.device)

        x_t, labels_t = self._apply_data_aug(x_t, labels_t, is_target_domain=True)
        y_e_t_processed = labels_t.get('event', y_e_t)

        # Combine data
        x_all = torch.cat([x_s, x_t], dim=0)
        d_all = torch.cat([d_s, d_t], dim=0)

        # Combine event tags
        y_event_all = None
        if y_e_s_processed is not None and y_e_t_processed is not None:
            if isinstance(y_e_s_processed, tuple) and isinstance(y_e_t_processed, tuple):
                y_event_all = (
                    torch.cat([y_e_s_processed[0], y_e_t_processed[0]], dim=0),
                    torch.cat([y_e_s_processed[1], y_e_t_processed[1]], dim=0)
                )
            elif isinstance(y_e_s_processed, tuple):
                y_event_all = (
                    torch.cat([y_e_s_processed[0], y_e_t_processed], dim=0),
                    torch.cat([y_e_s_processed[1], y_e_t_processed], dim=0)
                )
            elif isinstance(y_e_t_processed, tuple):
                y_event_all = (
                    torch.cat([y_e_s_processed, y_e_t_processed[0]], dim=0),
                    torch.cat([y_e_s_processed, y_e_t_processed[1]], dim=0)
                )
            else:
                y_event_all = torch.cat([y_e_s_processed, y_e_t_processed], dim=0)
        elif y_e_s_processed is not None:
            y_event_all = y_e_s_processed

        source_mask = torch.zeros(len(d_all), dtype=torch.bool, device=self.device)
        source_mask[:len(d_s)] = True

        return x_all, y_s_processed, y_event_all, d_all, source_mask

    def _prepare_features_for_classifier(self, features: torch.Tensor) -> torch.Tensor:
        """Prepare features for classifiers."""
        if isinstance(features, tuple):
            features = features[0]

        if not torch.is_tensor(features):
            raise TypeError(f"Expected features to be a tensor, got {type(features)}")

        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)
        elif features.dim() == 3:
            features = features.mean(dim=1)
        elif features.dim() == 2:
            if features.shape[1] < 32:
                pass

        if self.feature_normalize:
            features = F.normalize(features, p=2, dim=1)

        return features

    def _extract_backbone_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract scene logits and features from backbone."""
        if hasattr(self.backbone, 'forward_with_features'):
            output = self.backbone.forward_with_features(x)
            if isinstance(output, tuple) and len(output) == 2:
                scene_logits, features = output
            else:
                scene_logits = output
                features = self._extract_features_intercept(x)
                if features is None:
                    features = scene_logits
            return scene_logits, features

        if self.feature_layer:
            scene_logits = self.backbone(x)
            features = self._extract_features_from_layer(x, self.feature_layer)

            if isinstance(scene_logits, tuple):
                scene_logits = scene_logits[0]

            return scene_logits, features

        backbone_output = self.backbone(x)

        if isinstance(backbone_output, tuple):
            if len(backbone_output) >= 2:
                scene_logits = backbone_output[0]
                features = backbone_output[1]
            else:
                scene_logits = backbone_output[0]
                features = scene_logits
        else:
            scene_logits = backbone_output
            features = self._extract_features_intercept(x)
            if features is None:
                features = scene_logits

        return scene_logits, features

    def _extract_features_from_layer(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Extract features from a specific layer."""
        features = None
        hook_handle = None

        def hook_fn(module, input, output):
            nonlocal features
            features = output.clone()

        try:
            target_layer = dict(self.backbone.named_modules())[layer_name]
            hook_handle = target_layer.register_forward_hook(hook_fn)
            _ = self.backbone(x)
            hook_handle.remove()

            if isinstance(features, tuple):
                features = features[0]

            return features
        except Exception as e:
            if hook_handle is not None:
                hook_handle.remove()
            raise e

    def _extract_features_intercept(self, x: torch.Tensor) -> torch.Tensor:
        """Intercept features before classifier."""
        features = None

        def pre_classifier_hook(module, input):
            nonlocal features
            if isinstance(input, tuple):
                features = input[0].clone()
            else:
                features = input.clone()
            return input

        classifier = None
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Linear) and (name.endswith('classifier') or
                                                  name.endswith('fc') or
                                                  name.endswith('linear') or
                                                  name.endswith('head')):
                classifier = module
                break

        if classifier is not None:
            hook_handle = classifier.register_forward_pre_hook(pre_classifier_hook)
            _ = self.backbone(x)
            hook_handle.remove()

        return features

    def _apply_data_aug(self, x: torch.Tensor, labels: Dict[str, torch.Tensor],
                        is_target_domain: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply data augmentation."""
        sr = 32000

        if hasattr(self.data_aug, 'dir_aug') and 'device' in labels:
            x = self.data_aug.dir_aug(x, labels['device'], sr)

        if x.dim() == 4 and x.size(1) == 1:
            x_for_spec = x.squeeze(1)
        else:
            x_for_spec = x

        if self.spec_extractor:
            x_spec = self.spec_extractor(x_for_spec)
            if x_spec.dim() == 2:
                x = x_spec.unsqueeze(0).unsqueeze(0)
            elif x_spec.dim() == 3:
                x = x_spec.unsqueeze(1)
            elif x_spec.dim() == 4:
                x = x_spec
            else:
                raise ValueError(f"Unexpected spectrogram shape: {x_spec.shape}")
        else:
            x = x.unsqueeze(1) if x.dim() == 3 else x

        if hasattr(self.data_aug, 'mix_style'):
            x = self.data_aug.mix_style(x)

        if hasattr(self.data_aug, 'spec_aug'):
            x = self.data_aug.spec_aug(x)

        if hasattr(self.data_aug, 'mix_up') and not is_target_domain:
            available_keys = [k for k in self.mixup_label_keys if k in labels]
            if available_keys:
                mix_labels = [labels[k] for k in available_keys]
                x_mixed, labels_mixed = self.data_aug.mix_up([x], mix_labels)
                x = x_mixed[0]

                for i, k in enumerate(available_keys):
                    labels[k] = labels_mixed[i]

        return x, labels

    def _compute_dynamic_alpha(self, batch_idx: int) -> float:
        """Compute adaptive alpha for GRL."""
        if not self.adaptive_alpha:
            return self.grl_alpha

        if hasattr(self.trainer, 'num_training_batches') and self.trainer.num_training_batches:
            current_step = self.global_step
            total_steps = self.trainer.max_epochs * self.trainer.num_training_batches
            p = current_step / total_steps if total_steps > 0 else 0.0
        else:
            p = (self.current_epoch + batch_idx / 1000) / self.trainer.max_epochs

        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        return alpha * self.grl_alpha

    def _log_domain_specific_accuracies(self, domain_preds, d_all):
        """Log accuracy for each domain."""
        if self.mix_type_as_domain:
            for i, domain_name in enumerate(["source", "background-only", "known-event", "syth-unknown"]):
                mask = (d_all == i)
                if mask.any():
                    domain_specific_acc = (domain_preds[mask] == d_all[mask]).float().mean()
                    self.log(f'train/{domain_name}_domain_acc', domain_specific_acc, prog_bar=False)
        else:
            source_mask = (d_all == 0)
            target_mask = (d_all == 1)

            if source_mask.any():
                source_domain_acc = (domain_preds[source_mask] == d_all[source_mask]).float().mean()
                self.log('train/source_domain_acc', source_domain_acc, prog_bar=True)

            if target_mask.any():
                target_domain_acc = (domain_preds[target_mask] == d_all[target_mask]).float().mean()
                self.log('train/target_domain_acc', target_domain_acc, prog_bar=True)

    def _log_domain_distribution(self, d_all):
        """Log domain distribution."""
        if self.mix_type_as_domain:
            domain_counts = torch.bincount(d_all)
            total = len(d_all)
            for i, count in enumerate(domain_counts):
                if count > 0:
                    domain_name = ["source", "background-only", "known-event", "syth-unknown"][i]
                    ratio = count.float() / total
                    self.log(f'train/domain_{domain_name}_ratio', ratio, prog_bar=False)
        else:
            source_ratio = (d_all == 0).float().mean()
            target_ratio = (d_all == 1).float().mean()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int,
                        dataloader_idx: Optional[int] = None) -> torch.Tensor:
        """Validation step."""
        if isinstance(batch, dict) and 'source' in batch:
            x = batch['source']['wav']
            y_scene = batch['source'][self.class_label]
        else:
            x = batch['wav']
            y_scene = batch[self.class_label]

        # Process audio
        x_processed = self._process_audio_input(x)

        # Forward pass
        scene_logits = self.backbone(x_processed)

        if isinstance(scene_logits, tuple):
            scene_logits = scene_logits[0]

        scene_loss = F.cross_entropy(scene_logits, y_scene)
        preds = torch.argmax(scene_logits, dim=1)
        acc = (preds == y_scene).float().mean()

        self.log('val_loss_scene', scene_loss, prog_bar=True)
        self.log('val_acc_scene', acc, prog_bar=True)

        return scene_loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int,
                  dataloader_idx: Optional[int] = None) -> Dict[str, Any]:
        """Test step."""
        x = batch['wav']
        y_scene = batch[self.class_label]

        x_processed = self._process_audio_input(x)

        scene_logits = self.backbone(x_processed)

        if isinstance(scene_logits, tuple):
            scene_logits = scene_logits[0]

        scene_loss = F.cross_entropy(scene_logits, y_scene)
        scene_preds = torch.argmax(scene_logits, dim=1)
        acc = (scene_preds == y_scene).float().mean()

        output = {
            'y_true': y_scene,
            'y_pred': scene_preds,
            'logits': scene_logits,
            'loss': scene_loss,
            'accuracy': acc
        }

        self.log('test_loss_scene', scene_loss)
        self.log('test_acc_scene', acc)

        return output

    def _process_audio_input(self, x: torch.Tensor) -> torch.Tensor:
        """Process audio input."""
        while x.dim() > 3:
            x = x.squeeze(1)

        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)

        if self.spec_extractor:
            x_spec = self.spec_extractor(x)
        else:
            x_spec = x

        if x_spec.dim() == 2:
            x_spec = x_spec.unsqueeze(0).unsqueeze(0)
        elif x_spec.dim() == 3:
            x_spec = x_spec.unsqueeze(1)
        elif x_spec.dim() == 4:
            if x_spec.size(1) != 1:
                x_spec = x_spec.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected spectrogram shape: {x_spec.shape}")

        return x_spec

    def _move_to_device(self, device):
        """Move components to device."""
        self.backbone = self.backbone.to(device)

        if self.spec_extractor is not None:
            self.spec_extractor = self.spec_extractor.to(device)

        self.domain_grl = self.domain_grl.to(device)
        self.event_grl = self.event_grl.to(device)
        self.target_domain_grl = self.target_domain_grl.to(device)

        if self.domain_classifier is not None:
            self.domain_classifier = self.domain_classifier.to(device)

        if self.event_classifier is not None:
            self.event_classifier = self.event_classifier.to(device)

    def configure_optimizers(self):
        """Configure optimizer."""
        # Collect all parameters
        params = list(self.backbone.parameters())

        if self.domain_classifier is not None:
            params += list(self.domain_classifier.parameters())

        if self.event_classifier is not None:
            params += list(self.event_classifier.parameters())

        # Create optimizer
        optimizer = self.optimizer(params)

        def schedule_lambda(epoch: int):
            if epoch < self.warmup_len:
                return ((epoch + 1) / self.warmup_len) ** 2
            elif epoch >= self.warmup_len:
                decay_epochs = min(epoch - self.warmup_len, self.down_len)
                return max(1.0 - decay_epochs / self.down_len, self.min_lr)
            else:
                return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def _get_backbone_feature_dim(self) -> int:
        """
        Extract feature dimension from backbone by forward pass with dummy input.
        This method is called when classifiers need to be initialized before first training step.
        """
        if self._feature_dim is not None:
            return self._feature_dim

        try:
            # Create dummy input (batch_size=1, channels=1, freq=64, time=100)
            dummy_input = torch.randn(1, 1, 64, 100).to(self.device)
            _, features = self._extract_backbone_features(dummy_input)
            features = self._prepare_features_for_classifier(features)
            feature_dim = features.shape[1]
            self._feature_dim = feature_dim
            return feature_dim
        except Exception as e:
            print(f"[WARNING] Could not extract feature dimension from backbone: {e}")
            # Return a default dimension based on common backbones
            return 512  # Default for many CNN backbones