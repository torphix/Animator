import torch
import torchaudio
import torch.nn.functional as F
from collections import namedtuple
from modules.speech_embedding.speech_embedding import APCModel
    
PrenetConfig = namedtuple(
  'PrenetConfig', ['input_size', 'hidden_size', 'num_layers', 'dropout'])

RNNConfig = namedtuple(
  'RNNConfig',
  ['input_size', 'hidden_size', 'num_layers', 'dropout', 'residual'])


def load_pretrained_model(path='model_weights/apc_speech_emb.pt'):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    rnn_config = RNNConfig(input_size=80, hidden_size=512, num_layers=3,
                         dropout=0., residual=0)
    prenet_config = None
  
    model = APCModel(mel_dim=80, prenet_config=prenet_config,
                        rnn_config=rnn_config).cuda()
    model.load_state_dict(torch.load(path, map_location=device))
    return model
  
  
def prepare_inputs(wavs, mean_v=None, min_v=None, max_v=None, device=None):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    features, lengths = [], []
    max_len = 0
    for wav in wavs:
      feature = torchaudio.compliance.kaldi.fbank(
                            torch.tensor(wav, device=device).unsqueeze(0),
                            window_type='hamming',
                            use_energy=False,
                            dither=1,
                            num_mel_bins=80,
                            htk_compat=True)
      features.append(feature)
      max_len = max(max_len, feature.shape[0])
      lengths.append(torch.tensor(feature.shape[0]).int())

    # Pad
    pad_to = max_len - feature.shape[0]
    features = torch.stack(
                [F.pad(feature, (0, pad_to), value=0.0)
                for feature in features])
    if mean_v is not None:
      features = (features - mean_v) / (max_v - min_v)
    else:
      mean_v = torch.mean(features)
      max_v = torch.max(features)
      min_v = torch.min(features)
      features = (features - mean_v) / (max_v - min_v)
    return features, torch.stack(lengths).int()
    