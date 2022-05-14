import torch
import torch.nn as nn

from modules.wavenet import Wavenet

class Feature2Headpose(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_downsample = nn.Sequential(
            nn.Linear(config['in_d']*2, config['in_d']),
            nn.BatchNorm1d(config['in_d']),
            nn.LeakyReLU(0.2),
            nn.Linear(config['in_d'], config['in_d']))
        
        self.wavenet = Wavenet(**config['wavenet'])
        
    def train_step(self, audio_features, head_poses):
        '''
        First headpose frame should be all zeros
        '''
        future_frames = None
        audio_features = audio_features.reshape(-1, 512*2)
        n_frames = audio_features.shape[0] - future_frames
        predicted_head_poses = self.forward(audio_features, head_poses, n_frames)
        return predicted_head_poses
        
    def inference(self, audio_features):
        future_frames = None
        audio_features = audio_features.reshape(-1, 512*2)
        n_frames = audio_features.shape[0] - future_frames
        first_headpose = torch.zeros([n_frames, 12])
        predicted_head_poses = self.forward(audio_features, first_headpose, n_frames)
        return predicted_head_poses
    
    def forward(self, audio_features, head_poses, n_frames):
        '''
        Takes as input audio features then recursively
        computes next head pose frame by taking in audio
        features and adding previous head poses as conditions
        1st headpose is a zero frame
        '''
        BS, L, N = audio_features.shape
        audio_features = self.audio_downsample(audio_features.reshape(-1, N*2)).reshape(BS, L, -1)
        audio_features = audio_features.transpose(1,2)
        
        predicted_head_poses = []
        for i in range(n_frames):
            predicted_head_pose = self.wavenet(head_poses[i], audio_features)            
            predicted_head_poses.append(predicted_head_pose)
        return predicted_head_pose
        
        
    
        