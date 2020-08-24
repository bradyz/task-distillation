import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resnet


STEPS = 5


class SpatialSoftmax(torch.nn.Module):
    def forward(self, logit, temperature):
        """
        Assumes logits is size (n, c, h, w)
        """
        flat = logit.view(logit.shape[:-2] + (-1,))
        weights = F.softmax(flat / temperature, dim=-1).view_as(logit)

        x = (weights.sum(-2) * torch.linspace(-1, 1, logit.shape[-1]).type_as(logit)).sum(-1)
        y = (weights.sum(-1) * torch.linspace(-1, 1, logit.shape[-2]).type_as(logit)).sum(-1)

        return torch.stack((x, y), -1)


class Network(resnet.ResnetBase):
    def __init__(self, temperature, resnet_model, **resnet_kwargs):
        resnet_kwargs['input_channel'] = resnet_kwargs.get('input_channel', 3)

        super().__init__(resnet_model, **resnet_kwargs)

        self.temperature = temperature
        self.normalize = torch.nn.BatchNorm2d(resnet_kwargs['input_channel'])
        self.deconv = nn.Sequential(
                nn.BatchNorm2d(512), nn.ConvTranspose2d(512, 256, 3, 2, 1, 1), nn.ReLU(True),
                nn.BatchNorm2d(256), nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.ReLU(True),
                nn.BatchNorm2d(128), nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.ReLU(True),
                nn.BatchNorm2d(64), nn.Conv2d(64, STEPS, 1, 1, 0))
        self.extract = SpatialSoftmax()

    def forward(self, x):
        x = self.normalize(x)
        x = self.conv(x)
        x = self.deconv(x)

        return self.extract(x, self.temperature)
