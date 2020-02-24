import torch
import torch.nn.functional as F
from torchvision import models


class DenseNetWithDropout(models.DenseNet):
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        if self.training and self.dropout > 0:
            out = F.dropout(out, p=self.dropout)
        out = self.classifier(out)
        return out
