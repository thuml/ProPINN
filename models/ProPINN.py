import torch
import torch.nn as nn
import pdb
from util import get_perturbation
import numpy as np
import random


class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)


class Differential_perturbation(nn.Module):
    def __init__(self, range, size):
        super(Differential_perturbation, self).__init__()
        self.perturbation = get_perturbation(range=range, size=size).cuda()

    def forward(self, src):
        return src[:, None, :] + self.perturbation[None, :, :]


class Model(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layer, hidden_d_ff=64):
        super(Model, self).__init__()
        self.linear_emb = nn.Sequential(*[
            nn.Linear(in_dim, hidden_dim // 4),
            WaveAct(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        ])
        self.multiregion_mixer = nn.Sequential(*[
            nn.Linear(4, 8),
            WaveAct(),
            nn.Linear(8, 1),
        ])
        layers = []
        for i in range(num_layer):
            if i == 0:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_d_ff))
                layers.append(WaveAct())
            else:
                layers.append(nn.Linear(in_features=hidden_d_ff, out_features=hidden_d_ff))
                layers.append(WaveAct())
        layers.append(nn.Sequential(*[
            nn.Linear(in_features=hidden_d_ff, out_features=hidden_d_ff),
            WaveAct(),
            nn.Linear(in_features=hidden_d_ff, out_features=out_dim),
        ]))
        self.linear_out = nn.Sequential(*layers)
        self.region_perturbation_1 = Differential_perturbation(range=0.01, size=3)
        self.region_perturbation_2 = Differential_perturbation(range=0.05, size=5)
        self.region_perturbation_3 = Differential_perturbation(range=0.09, size=7)

    def forward(self, x, t):
        src = torch.cat((x, t), dim=-1)
        # Differential perturbation
        src_1 = self.linear_emb(src)
        src_2 = self.linear_emb(self.region_perturbation_1(src)).mean(dim=1)
        src_3 = self.linear_emb(self.region_perturbation_2(src)).mean(dim=1)
        src_4 = self.linear_emb(self.region_perturbation_3(src)).mean(dim=1)
        # Multi-region mixing
        src = torch.stack([src_1, src_2, src_3, src_4], dim=1)
        e_outputs = self.multiregion_mixer(src.permute(0, 2, 1).contiguous())[:, :, 0]
        output = self.linear_out(e_outputs)
        return output
