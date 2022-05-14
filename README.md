1) Seperate content and identity from speech
    - AutoVC Encoder extracts content
2) Seperate training process for each module using different datasets
    - Wav to landmark
    - Landmark + id frame to image 


Speaker Embedding Training:
    - Dataset: VCTK
    - Initialize pretrained
    - Generate content encodings of each utterance
    - Generate speaker embedding of same speaker but different utterance 
    - Pass both content encoding + speaker embedding -> decoder -> Reconstruct original audio (spectrogram) 

Speech Content Animation Training:  
    - Dataset: Obama Weekly
    - 

Speaker Aware Animation Training:
    - Dataset VoxCeleb2, filtered down to 1232 of 67 speakers selection: accurate landmark detection


Head Pose And Upper Body motion synthesis:
    - Model predicts the mean and standard deviation for translation and 3D rotation of previous head landmarks
    - Inputs: Previous head landmarks + Audio -> Neural network -> Mean and STD of next translation and rotation of each landmark

# References:
@inproceedings{chung2019unsupervised,
  title = {An unsupervised autoregressive model for speech representation learning},
  author = {Chung, Yu-An and Hsu, Wei-Ning and Tang, Hao and Glass, James},
  booktitle = {Interspeech},
  year = {2019}
}
@inproceedings{chung2020generative,
  title = {Generative pre-training for speech with autoregressive predictive coding},
  author = {Chung, Yu-An and Glass, James},
  booktitle = {ICASSP},
  year = {2020}
}



Audio -> Mouth pose 
Audio + Previous headpose -> Headpose 

Mouth pose + Headpose + Randomly sampled blinks and eyebrows -> Combine into frame

Pose frame -> Pix2Pix

Predicting easier intermediates such as facial landmarks is often prefereable for solving problems