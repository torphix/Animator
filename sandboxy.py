import torch
import librosa
from modules.speech_embedding.manifold_projection import ManifoldProjection
from modules.speech_embedding.utils import (prepare_inputs, 
                                            load_pretrained_model)

wav, sr = librosa.load('/home/j/Desktop/Programming/animator/data/1_jocko.wav', sr=16000)
inputs, lengths = prepare_inputs([wav])
model = load_pretrained_model()
_, feat = model(inputs, lengths)


manifold_projection = ManifoldProjection('/home/j/Desktop/Programming/animator/APC_feature_base.pt', 0.8, K=10)
feat = manifold_projection(feat[-1])
print(feat.shape)


# Next audio to mouth related motion