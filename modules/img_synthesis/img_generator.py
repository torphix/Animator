import cv2
import torch
import torch.nn as nn
from modules.unet import Unet
from utils import draw_landmarks


class ImgSynthesis(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.unet = Unet(config['hid_ds'], 
                         config['sizes'],
                         config['stride'], 
                         config['upsample_type'])
        
    def forward(self, sample_images, landmarks):
        # Add Audio?
        landmarks = draw_landmarks([*sample_images.shape[1:], 1], landmarks) / 255
        x = torch.cat((sample_images, landmarks), dim=1) # Channel wise concatenation
        out = self.unet(x)
        return out