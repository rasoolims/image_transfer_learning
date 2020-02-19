import torch
import torch.nn.functional as F
from torchvision import models


class ResnetWithDropout(models.ResNet):
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.training and self.dropout > 0:
            x = F.dropout(x, p=self.dropout)

        x = self.fc(x)

        return x
