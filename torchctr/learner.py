from dataclasses import dataclass

import torch
from torch import nn, optim
from .datasets.utils import defaults, totensor


@dataclass
class Learner:
    model: nn.Module = model.to(defaults.device)
    criterion: nn.Module
    opt: optim.Optimizer

    def fit(input_loader, epoch=100):
        pass
    
    @torch.no_grad()
    def predict(input):
        pass
    
    def save_trained_model(self, path):
    """save trained model's weights.
    Args:
        path (str): the path to save checkpoint.
    """

    # save model weights
    torch.save(self.model.state_dict(), path)

    def save_model(self, path):
        """save model.
        Args:
            path (str): the path to save checkpoint.
        """

        # save model weights
        torch.save(self.model, path)
