import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn


class ManifoldProjection(nn.Module):
    '''
    Input: Extracted speech representation [BS, L, N]
    Computation: 
        - Find the K nearest datapoints between input and feature database
        - Compute barycentric coordinates between nearest neighbours and input
        - Project the input speech representation to the new barycentric coordinates
    Output: Input projected to be close to the K nearest neighbours
    Methods taken & adapted from: MIT License
    https://github.com/YuanxunLu/LiveSpeechPortraits
    '''
    def __init__(self, feature_database, LLE_percent, K=10, device=None):
        super().__init__()
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else: self.device = device
        self.feature_database = torch.load(feature_database, self.device)
        self.LLE_percent = LLE_percent
        self.K = K
        
    @staticmethod
    def KNN_torch(features, feature_database, K):
        # Training
        feat_base_norm = (feature_database ** 2).sum(-1)
        feats_norm = (features ** 2).sum(-1)
        diss = (feats_norm.view(-1, 1)
                + feat_base_norm.view(1, -1)
                - 2 * features @ feature_database.t()  # Rely on cuBLAS for better performance!
            )
        ind = diss.topk(K, dim=1, largest=False).indices
        return ind.squeeze(0).cpu().numpy()
    
    @staticmethod
    def LLE_projection(audio_features, feature_database, ind, audio_len, device):
        nframe = audio_features.shape[0]
        feat_fuse = torch.zeros_like(audio_features)
        w = torch.zeros([nframe, ind.shape[1]])
        for i in tqdm(range(nframe), desc='LLE projection'):
            current_K_feats = feature_database[ind[i]]
            w[i], feat_fuse[i] = ManifoldProjection.solve_LLE_projection(audio_features[i], current_K_feats, device)
        return w, feat_fuse
    
    @staticmethod
    def solve_LLE_projection(feat, feat_base, device):
        '''find LLE projection weights given feat base and target feat
        Args:
            feat: [ndim, ] target feat
            feat_base: [K, ndim] K-nearest feat base
        =======================================
        We need to solve the following function
        ```
            min|| feat - \sum_0^k{w_i} * feat_base_i ||, s.t. \sum_0^k{w_i}=1
        ```
        equals to:
            ft = w1*f1 + w2*f2 + ... + wk*fk, s.t. w1+w2+...+wk=1
            = (1-w2-...-wk)*f1 + w2*f2 + ... + wk*fk
        ft-f1 = w2*(f2-f1) + w3*(f3-f1) + ... + wk*(fk-f1)
        ft-f1 = (f2-f1, f3-f1, ..., fk-f1) dot (w2, w3, ..., wk).T
            B  = A dot w_,  here, B: [ndim,]  A: [ndim, k-1], w_: [k-1,]
        Finally,
        ft' = (1-w2-..wk, w2, ..., wk) dot (f1, f2, ..., fk)
        =======================================    
        Returns:
            w: [K,] linear weights, sums to 1
            ft': [ndim,] reconstructed feats
        '''
        K, ndim = feat_base.shape
        if K == 1:
            feat_fuse = feat_base[0]
            w = torch.tensor([1], device=device)
        else:
            w = torch.zeros(K, device=device)
            B = feat - feat_base[0]   # [ndim,]
            A = (feat_base[1:] - feat_base[0]).T   # [ndim, K-1]
            AT = A.T
            w[1:] = torch.linalg.solve(torch.matmul(AT, A), torch.matmul(AT, B))
            w[0] = 1 - w[1:].sum()
            feat_fuse = torch.matmul(w, feat_base)
        return w, feat_fuse
    
    @staticmethod
    def project(audio_features, feature_fuse, LLE_percent):
        return audio_features * (1 - LLE_percent) + feature_fuse * LLE_percent
    
    def forward(self, audio_features):
        if len(audio_features.shape) == 3:
            audio_features = audio_features.squeeze(0)
        ind = ManifoldProjection.KNN_torch(audio_features, self.feature_database, self.K)
        weights, feature_fuse = ManifoldProjection.LLE_projection(audio_features, self.feature_database,
                                                                  ind, audio_features.shape[0], device=self.device)
        audio_features = ManifoldProjection.project(audio_features, feature_fuse, self.LLE_percent)
        return audio_features