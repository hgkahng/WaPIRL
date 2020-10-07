# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    Implementation of cross entropy loss with label smoothing.
    Follows the implementation of the two followings:
        https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
        https://github.com/pytorch/pytorch/issues/7455#issuecomment-513735962
    """
    def __init__(self, num_classes, smoothing=.0, dim=1, reduction='mean', class_weights=None):
        """
        Arguments:
            num_classes: int, specifying the number of target classes.
            smoothing: float, default value of 0 is equal to general cross entropy loss.
            dim: int, aggregation dimension.
            reduction: str, default 'mean'.
            class_weights: 1D tensor of shape (C, ) or (C, 1).
        """
        super(LabelSmoothingLoss, self).__init__()
        assert 0 <= smoothing < 1
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim

        assert reduction in ['sum', 'mean']
        self.reduction = reduction

        self.class_weights = class_weights

    def forward(self, pred: torch.FloatTensor, target: torch.LongTensor):
        """
        Arguments:
            pred: 2D torch tensor of shape (B, C)
            target: 1D torch tensor of shape (B, )
        """
        pred = F.log_softmax(pred, dim=self.dim)
        true_dist = self.smooth_one_hot(target, self.num_classes, self.smoothing)
        multiplied = -true_dist * pred

        if self.class_weights is not None:
            weights = self.class_weights.to(multiplied.device)
            summed = torch.matmul(multiplied, weights.view(self.num_classes, 1))  # (B, C) @ (C, 1) -> (B, 1)
            summed = summed.squeeze()                                             # (B, 1) -> (B, )
        else:
            summed = torch.sum(multiplied, dim=self.dim)                          # (B, C) -> sum -> (B, )

        if self.reduction == 'sum':
            return summed
        elif self.reduction == 'mean':
            return torch.mean(summed)
        else:
            raise NotImplementedError

    @staticmethod
    def smooth_one_hot(target: torch.LongTensor, num_classes: int, smoothing: float = 0.):
        assert 0 <= smoothing < 1
        confidence = 1. - smoothing
        label_shape = torch.Size((target.size(0), num_classes))
        with torch.no_grad():
            true_dist = torch.zeros(label_shape, device=target.device)
            true_dist.fill_(smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), confidence)
        return true_dist  # (B, C)



class WaPIRLLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super(WaPIRLLoss, self).__init__()
        self.temperature = temperature
        self.similarity_1d = nn.CosineSimilarity(dim=1)
        self.similarity_2d = nn.CosineSimilarity(dim=2)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(self,
                anchors: torch.Tensor,
                positives: torch.Tensor,
                negatives: torch.Tensor):

        assert anchors.size() == positives.size()
        batch_size, _ = anchors.size()
        num_negatives, _ = negatives.size()

        negatives = negatives.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, numN, F)
        negatives = negatives.detach()

        # Similarity between queries & positive samples
        sim_a2p = self.similarity_1d(anchors, positives)             # (B, )
        sim_a2p = sim_a2p.div(self.temperature).unsqueeze(1)         # (B, 1)

        # Similarity between positive & negative samples
        sim_a2n = self.similarity_2d(
            positives.unsqueeze(1).repeat(1, num_negatives, 1),      # (B, numN, F)
            negatives                                                # (B, numN, F)
        )
        sim_a2n = sim_a2n.div(self.temperature)                      # (B, numN)

        # Get class logits
        logits = torch.cat([sim_a2p, sim_a2n], dim=1)                # (B, 1 + numN)

        # Get cross entropy loss
        loss = self.cross_entropy(
            logits,                                                  # (B, 1 + numN)
            torch.zeros(logits.size(0)).long().to(logits.device)     # (B, )
        )

        return loss, logits.detach()
