# Adapted from: https://github.com/fschmid56/cpjku_dcase23/tree/main
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


class ExpWarmupLinearDown(LambdaLR):
    def __init__(self, optimizer, warmup_len, down_len, down_start, min_lr, last_epoch=-1):
        lambda_fn = exp_warmup_linear_down(warmup_len, down_len, down_start, min_lr)
        super().__init__(optimizer, lambda_fn, last_epoch)


def exp_warmup_linear_down(warmup, rampdown_length, start_rampdown, last_value):
    """
    Simple learning rate scheduler. This function returns the factor the maximum
     learning rate is multiplied with. It includes:
    1. Warmup Phase: lr exponentially increases for 'warmup' number of epochs (to a factor of 1.0)
    2. Constant LR Phase: lr reaches max value (factor of 1.0)
    3. Linear Decrease Phase: lr decreases linearly starting from epoch 'start_rampdown'
    4. Finetuning Phase: phase 3 completes after 'rampdown_length' epochs, followed by a finetuning phase using
                        a learning rate of max lr * 'last_value'
    """
    rampup = exp_rampup(warmup)
    rampdown = linear_rampdown(rampdown_length, start_rampdown, last_value)
    def wrapper(epoch):
        return rampup(epoch) * rampdown(epoch)
    return wrapper


def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def wrapper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.5, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return wrapper


def linear_rampdown(rampdown_length, start=0, last_value=0):
    def wrapper(epoch):
        if epoch <= start:
            return 1.
        elif epoch - start < rampdown_length:
            return last_value + (1. - last_value) * (rampdown_length - epoch + start) / rampdown_length
        else:
            return last_value
    return wrapper


class LinearWarmupCosineDown(LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps, num_cycles, last_epoch=-1):
        lambda_fn = linear_warmup_cosine_down(warmup_steps, total_steps, num_cycles)
        super().__init__(optimizer, lambda_fn, last_epoch)


def linear_warmup_cosine_down(warmup_steps, total_steps, num_cycles=0.5):
    """
    Cosine learning rate schedule with linear warmup.
    This function returns a wrapper(epoch) → lr_factor, similar to exp_warmup_linear_down.

    Args:
        warmup_steps (int): Number of warmup steps (linear ramp up from 0 to 1).
        total_steps (int): Total number of training steps.
        num_cycles (float): Number of cosine cycles in the decay phase. Default 0.5 = half cosine (standard HF).

    Returns:
        wrapper(epoch): function that returns LR factor for the given epoch/step.
    """
    def wrapper(step):
        # 1. Linear warmup
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        # 2. Cosine decay after warmup
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(progress, 1.0)  # clamp

        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * num_cycles * 2.0 * progress))
        return cosine_decay

    return wrapper