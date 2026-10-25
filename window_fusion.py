import torch
import torch.nn as nn

class WindowFusionMean(nn.Module):

    def __init__(self, heads):
        super().__init__()
        self.obj_head = heads.obj_head
        self.delta_head = heads.delta_head
        self.sigma_head = heads.sigma_head

    def forward(self, feat):
        z = feat.mean(1)                                    # [B,M,G,d] -> [B,G,d] 
        obj = self.obj_head(z).squeeze(-1)                  # [B,G]
        delta = 0.5 * torch.tanh(self.delta_head(z).squeeze(-1))
        log_sigma = self.sigma_head(z).squeeze(-1)          
        return obj, delta, log_sigma
