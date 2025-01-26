"""
Code adapted from https://lightning.ai/lightning-ai/studios/code-lora-from-scratch and https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
"""
import torch
import torch.nn.functional as F


class LoRALayer():
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        self.rank = rank
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.lora_A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        
        self.alpha = alpha
        self.scaling = self.alpha / self.rank

        self.dropout = torch.nn.Dropout(p=dropout)

        self.merged = False


class LinearWithLoRA(torch.nn.Linear, LoRALayer):
    def __init__(
        self,
        linear,
        rank: int,
        alpha: float,
        dropout: float,
        **kwargs
    ):
        torch.nn.Linear.__init__(self, linear.in_features, linear.out_features, **kwargs)
        LoRALayer.__init__(
            self,
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        self.weight = linear.weight
        self.bias = linear.bias
        self.training = True

    def forward(self, x):
        linear_out = F.linear(x, self.weight, bias=self.bias)
        lora_out = self.scaling * (self.dropout(x) @ self.lora_A @ self.lora_B)
        return linear_out + lora_out
