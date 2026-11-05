import torch
from torchvision import transforms, models
from infoseclab.models.resnet import ResNet18
from infoseclab import common


class Defense(object):
    """
    An abstract class for a defense.
    """

    def __init__(self, device):
        self.device = device

    def get_logits(self, x):
        """
        Get the logits of the defense on a batch of images.
        :param x: the batch of images of size (batch_size, 3, 32, 32) in the range [0, 255]
        :return: the logits of the defense of size (batch_size, 10)
        """
        raise NotImplementedError()

    def classify(self, x):
        """
        Get the classification of the defense on a batch of images.
        :param x: the batch of images of size (batch_size, 3, 32, 32) in the range [0, 255]
        :return: the classification of the defense of size (batch_size,)
        """
        return torch.argmax(self.get_logits(x), dim=-1)


class ResNet(Defense):
    """
    A defense that uses a ResNet18 model trained on CIFAR10.
    """
    def __init__(self, device):
        super().__init__(device)
        self.model = ResNet18()
        # Load pre-trained weights
        checkpoint = torch.load(common.get_checkpoint_abs_path("data/resnet.pth"), map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.to(device)
        self.model.eval()
        # CIFAR10 normalization
        self.normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                              std=[0.2023, 0.1994, 0.2010])

    def get_logits(self, x):
        return self.model(self.normalize(x / 255.0))
