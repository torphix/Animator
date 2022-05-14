import torch
import torch.nn as nn


class ImgSynthesisLoss(nn.Module):
    def __init__(self):
        self.l1_loss = nn.L1Loss()
        
    def forward(self, outputs, targets):
        loss = self.l1_loss(outputs, targets)
        return loss