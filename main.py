import os
import sys
import yaml
import argparse
import subprocess

from data.process import ProcessDataset
from utils import open_configs


if __name__ == '__main__':
    command = sys.argv[1]    
    parser = argparse.ArgumentParser()

    # Dataset Commands
    if command == 'process_dataset':
        data_config = open_configs(['configs/data.yaml'])[0]
        data_processor = ProcessDataset(data_config)
        data_processor.process_dataset()
        
    elif command == 'draw_landmarks':
        video_path = parser.add_argument('-v', '--video_path')
        landmarks_path = parser.add_argument('-l', '--landmarks_path')
        args, leftover_args = parser.parse_known_args()
        data_config = open_configs(['configs/data.yaml'])[0]
        processor = ProcessDataset(data_config)
        processor.draw_landmarks_on_video(args.video_path, args.landmarks_path)
        
    else:
        print(f'''
              Command {command} not found, try:
                - process_dataset
                - draw_landmarks
              ''')