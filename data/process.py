import os
import torch
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip, ImageSequenceClip

import cv2
import dlib
from imutils import face_utils

from utils import draw_landmarks

class Processor:
    def __init__(self, data_config, face_landmarks, device=None):
        '''
        Images are extracted from video and aligned to face_landmarks
        
        '''
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.input_dir = data_config['input_dir']
        self.img_size = data_config['img_size']


    def process_image(self, img):
        pass
            
    def process_video(self, video):
        pass
    
    
class ProcessDataset:
    def __init__(self, data_config, base_image=None, device=None):
        '''
        Landmarks are extracted from base_image and then all
        subsequent video are aligned to the base images
        If none is passed as base image image from first video
        in dataset is used for all subsequent video alignment
        Dataset: 
            - Mel Spectrogram
            - Headpose and shoulder landmarks for each frame
            - Mouth landmarks for each frame
            - Processed images for each frame
        '''
        self.input_dir = f"data/{data_config['input_dir']}"
        self.img_size = data_config['img_size']
        self.face_anchor_points = [33, 36, 39, 42, 45]
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            "model_weights/shape_predictor_68_face_landmarks.dat")
        self.base_image = base_image

    def process_dataset(self):
        os.makedirs(f'{self.input_dir}/landmarks', exist_ok=True)
        
        for i, video_name in enumerate(tqdm(os.listdir(f'{self.input_dir}/video'), 
                                       desc='Processing dataset')):
            video_path = f'{self.input_dir}/video/{video_name}'
            if video_path.endswith('mp4') == False: continue
            self.resize_video(video_path)
            video = cv2.VideoCapture(video_path)
            landmarks = self.extract_landmarks(video)
            torch.save(landmarks, f'{self.input_dir}/landmarks/{video_name}.pt')
            
    def resize_video(self, video_path):
        video = cv2.VideoCapture(video_path)
        _, frame = video.read()
        if frame.shape[0:1] == self.img_size: 
            return
        # Find the landmarks for first frame
        while True:
            not_done, img = video.read()
            if not_done == False:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bbs = self.detector(img, 0)
            if len(bbs) == 0:
                print(f'No landmarks found in {video_path}')
                os.remove(video_path)
                return
            for bb in bbs:
                shape = self.predictor(img, bb)
                landmarks = face_utils.shape_to_np(shape)
            avg_y = int(sum([lm[0] for lm in landmarks])/len(landmarks))
        # Crop frames
        video = VideoFileClip(video_path)
        new_frames = []
        for frame in video.iter_frames():
            # Edit X
            if frame.shape[0] != self.img_size[0]:
                # As width is > than height take diff 
                x_diff = frame.shape[0] - self.img_size[0]
                frame = frame[:frame.shape[0]-x_diff,:] 
            # Edit Y
            if frame.shape[1] != self.img_size[1]:
                frame = frame[:, avg_y-self.img_size[0]:avg_y+self.img_size[1]]
            new_frames.append(frame)
        new_video = ImageSequenceClip(new_frames, fps=video.fps)
        new_video.write_videofile(video_path, fps=video.fps)

    def extract_landmarks(self, video):
        landmarks = []
        while True:
            not_done, img = video.read()
            
            if not_done == False:
                return landmarks

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bbs = self.detector(img, 0)
            for bb in bbs:
                shape = self.predictor(img, bb)
                shape = face_utils.shape_to_np(shape)
                landmarks.append(shape)
                            
    def draw_landmarks_on_video(self, video_path, landmarks_path):
        landmarks = np.array(torch.load(landmarks_path))
        video = VideoFileClip(video_path)
        frames = [frame for frame in video.iter_frames()][:landmarks.shape[0]]
        new_frames = []
        for i,frame in enumerate(frames):
            for landmark in landmarks[i]:
                cv2.circle(frame, (landmark[0], landmark[1]), 1, (255,0,0,), 1)   
            landmark_mask = draw_landmarks([*frame.shape], landmarks[i])
            img = np.where(landmark_mask != 0, landmark_mask, frame)
            new_frames.append(img)
        
        new_video = ImageSequenceClip(new_frames, fps=video.fps)
        new_video.write_videofile('output.mp4', fps=video.fps)