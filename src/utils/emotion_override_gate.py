import torch
from torch import nn



class EmotionOverrideGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.override_net = nn.Sequential(
            nn.LayerNorm(dim * 2 + 2),
            nn.Linear(dim * 2 + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, text_encoding, profile_vec, self_ref_score, positive_score):
        batch_size = text_encoding.size(0)

        # Flatten everything
        x = torch.cat([
            text_encoding,
            profile_vec,
            self_ref_score.view(batch_size, 1),
            positive_score.view(batch_size, 1)
        ], dim=-1)

        return self.override_net(x)
