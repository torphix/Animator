import cv2
import yaml
import numpy as np

def open_configs(configs:list) -> list:
    open_configs = []
    for config in configs:
        with open(config, 'r') as f:
            open_configs.append(yaml.load(f.read(), Loader=yaml.FullLoader))
    return open_configs

def draw_landmarks(img_size, landmarks):
    '''
    Draws and connects associated landmarks
    img_size should also include channels
    '''
    prev_landmark = 0
    left_mouth_edge, right_mouth_edge = 48, 55
    img = np.zeros(img_size)
    color = (255,0,0) if img_size[-1] == 3 else (255)
    for i, landmark in enumerate(landmarks):
        if i in range(0, 17): # Jaw
            if i != 0:
                cv2.line(img, prev_landmark, landmark, color, 1, 1)
        elif i in range(17, 22): # Left eye brow
            if i != 17:
                cv2.line(img, prev_landmark, landmark, color, 1, 1)
        elif i in range(22, 27): # Right eye brow
            if i != 22:
                cv2.line(img, prev_landmark, landmark, color, 1, 1)
        elif i in range(27, 31): # Nose bridge
            if i != 27:
                cv2.line(img, prev_landmark, landmark, color, 1, 1)
        elif i in range(31, 36): # Nose base
            if i != 31:
                cv2.line(img, prev_landmark, landmark, color, 1, 1)
        elif i in range(36, 42): # Left eye
            if i == 36: left_eye_corner = landmark
            else:
                cv2.line(img, prev_landmark, landmark, color, 1, 1)
            if i == 41: 
                cv2.line(img, landmark, left_eye_corner, color, 1, 1)
        elif i in range(42, 48): # Right eye
            if i == 42: right_eye_corner = landmark
            else:
                cv2.line(img, prev_landmark, landmark, color, 1, 1)
            if i == 47:
                cv2.line(img, landmark, right_eye_corner, color, 1, 1)
                
        # Pick up mouth from here
        elif i in range(48, 68):
            if i == 48: left_mouth_edge = landmark
            elif i == 54: right_mouth_edge = landmark
            if i in range(48, 60): # Outer mouth
                if i != 48:
                    cv2.line(img, prev_landmark, landmark, color, 1, 1)
                if i == 59:
                    cv2.line(img, left_mouth_edge, landmark, color, 1, 1)
            elif i in range(60, 68):
                if i == 60:
                    cv2.line(img, left_mouth_edge, landmark, color, 1, 1)
                elif i == 64:
                    cv2.line(img, right_mouth_edge, landmark, color, 1, 1)
                    cv2.line(img, prev_landmark, landmark, color, 1, 1)
                elif i == 67:
                    cv2.line(img, landmark, left_mouth_edge, color, 1, 1)
                    cv2.line(img, prev_landmark, landmark, color, 1, 1)
                else:
                    cv2.line(img, prev_landmark, landmark, color, 1, 1)
        prev_landmark = landmark
    return img