import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Implementation of the Focal Loss for binary classification, particularly useful for handling 
    class imbalance and improving robustness to hard or ambiguous examples.

    This loss modifies the standard Binary Cross-Entropy (BCE) loss by applying a modulating factor 
    to down-weight easy examples and focus learning on hard, misclassified ones.

    Parameters
    ----------
    alpha : float, optional (default=0.25)
        Balancing factor for the positive class. Helps address class imbalance.

    gamma : float, optional (default=2.0)
        Focusing parameter. A higher gamma increases the down-weighting of easy examples.

    reduction : str, optional (default='mean')
        Specifies the reduction to apply to the output:
        - 'mean': return the mean of the loss over the batch
        - 'sum': return the sum of the loss over the batch
        - 'none': return the loss for each sample individually

    Example
    -------
    >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    >>> inputs = torch.sigmoid(torch.randn(8))  # predicted probabilities
    >>> targets = torch.randint(0, 2, (8,)).float()  # ground truth
    >>> loss = loss_fn(inputs, targets)
    >>> print(loss.item())

    Notes
    -----
    - This implementation assumes that `inputs` are probabilities (i.e., after sigmoid).
    - Commonly used in tasks like object detection or deepfake detection where class imbalance is significant.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
