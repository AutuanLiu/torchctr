from dataclasses import dataclass

import torch
from torch import nn, optim


@dataclass
class Learner:
    model: nn.Module
    criterion: nn.Module
    opt: optim.Optimizer

    def fit(input, epoch=100):
        pass
    
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
