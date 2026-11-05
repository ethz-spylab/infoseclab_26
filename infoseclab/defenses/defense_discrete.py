import torch
import numpy as np
import torch.nn as nn
from infoseclab.defenses.defense import Defense
from infoseclab import common
from infoseclab.defenses.defense import ResNet18

N_BINS = 200

class ResNetDiscrete(Defense):
    """
    A defense that uses a ResNet18 model on discretized features
    """
    def __init__(self, device, model_path=common.get_checkpoint_abs_path("data/resnet_discrete.pth")):
        super().__init__(device)
        self.model = ResNet18()
        self.model.conv1 = nn.Conv2d(3*N_BINS, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.model.to(device)

    def encode(self, x):
        assert len(x.shape) == 4
        assert x.shape[1:] == (3, 32, 32)
        assert x.min() >= 0 and x.max() <= 1.0

        # put the channels last
        x = x.permute(0, 2, 3, 1)
        shape = list(x.shape)
        
        # add a dummy dimension
        x = x.unsqueeze(-1)

        # discretize into 20 bins
        thresholds = torch.from_numpy(np.arange(0, 1, 1/N_BINS)).to(x.device).float()
        gt_threshold = x > thresholds
        x = gt_threshold.float()

        # reshape to original shape with 20x channels
        shape[-1] *= thresholds.shape[0]
        x = x.reshape(shape)
        x = x.permute(0, 3, 1, 2)

        assert x.shape[1:] == (3*N_BINS, 32, 32)
        return x

    def get_logits(self, x):
        return self.model(self.encode(x / 255.0))
