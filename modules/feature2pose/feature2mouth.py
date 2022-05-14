import torch
import torch.nn as nn


class Feature2Mouth(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sr = config['sr']
        self.fps = config['fps']

        self.lstm = nn.LSTM(**config['lstm'])
        self.mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, config['output_size']))

    def forward(self, audio_features):
        '''Predicts the 3D displacements of each mouth landmark'''
        output, (h0, c0) = self.lstm(audio_features)
        output = self.mlp(output)
        
        return output