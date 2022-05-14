import os
import torch
import random
from torchvision import transforms
from torch.utils.data import Dataset
from moviepy.editor import VideoFileClip


class ImgSynthesisDataset(Dataset):
    def __init__(self, config):
        self.input_dir = config['input_dir']
        self.file_names = [file for file in os.listdir(f'{self.input_dir}/video/')]
        
        self.img_transform = transforms.Compose([
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        
    def read_video(self, video_path):
        video = torch.stack([frame for frame in VideoFileClip(video_path).iter_frames()])
        return video
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        video_path = f'{self.input_dir}/video/{self.file_names[idx]}'
        landmark_path = f'{self.input_dir}/landmarks/{self.file_names[idx]}.pt'
        
        target_images = self.read_video(video_path)
        sample_images = target_images[random.sample(range(target_images.shape[0]), 4)]
        landmark_path = torch.load(landmark_path)
        
        return sample_images, landmark_path, target_images
        