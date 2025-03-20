import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def smoothed_loss(predictions, targets, device, smoothing=0.1):
    """
    Compute the cross-entropy loss with label smoothing.

    Args:
        predictions (Tensor): Predictions from the model (logits).
        targets (Tensor): Ground truth labels.
        device (torch.device): Device to perform computation.
        smoothing (float): Smoothing factor (0 means no smoothing).

    Returns:
        Tensor: Computed loss.
    """
    num_classes = predictions.size(1)
    one_hot_targets = F.one_hot(targets, num_classes).float().to(device)
    one_hot_targets = one_hot_targets.permute(0, 3, 1, 2)
    smooth_targets = one_hot_targets * (1 - smoothing) + smoothing / num_classes
    log_probs = F.log_softmax(predictions, dim=1)

    return -(smooth_targets * log_probs).sum(dim=1).mean()

def cross_entropy_2d(predict, target):
    """
    Args:
        predict:(n, c, h, w)
        target:(n, h, w)
    """
    assert not target.requires_grad
    assert predict.dim() == 4
    assert target.dim() == 3
    assert predict.size(0) == target.size(0), f"{predict.size(0)} vs {target.size(0)}"
    assert predict.size(2) == target.size(1), f"{predict.size(2)} vs {target.size(1)}"
    assert predict.size(3) == target.size(2), f"{predict.size(3)} vs {target.size(3)}"
    n, c, h, w = predict.size()
    target_mask = (target >= 0) * (target != 255)
    target = target[target_mask]
    if not target.data.dim():
        return Variable(torch.zeros(1))
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    loss = F.cross_entropy(predict, target, size_average=True)
    return loss

def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))
