import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, config):
        
        
# Create patch discriminator & custom training loop
# Discriminator should be updated a few more times than generator 