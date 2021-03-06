import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    I got this from https://github.com/adambielski/siamese-triplet/blob/master/losses.py#L24
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin: float = 1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, size_average: bool = True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        anchor_view = anchor.view(anchor.size(0), 1, -1)
        distance_negative = (anchor_view - negative).pow(2).sum(2)  # .pow(.5)
        distance_negative_min = distance_negative.min(1)[0] # Finding the hardest negative sample
        losses = F.relu(distance_positive - distance_negative_min + self.margin)
        return losses.mean() if size_average else losses.sum()
