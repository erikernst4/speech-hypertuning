from typing import List, Optional

import torch

from .attention_poolings import (AttentionPooling, PositionalEncoding,
                                 SelfAttentionLayer)


class WeightedAverageLayerPooling(torch.nn.Module):
    def __init__(self, upstream_layers_output_to_use: List[int]):
        super().__init__()
        self.upstream_layers_output_to_use = upstream_layers_output_to_use

        self.avg_weights = torch.nn.Parameter(
            torch.ones(
                len(upstream_layers_output_to_use),
            )
        )

    def forward(self, x):
        """
        Calculate weighted average of the layers embeddings.

        Args:
            x (torch.Tensor): Input tensor (#batch_size, #upstream_layers, upstream_hidden_dim).
        """
        # x shape (batch_size, upstream_layer, upstream_hidden_dim)
        w = torch.nn.functional.softmax(self.avg_weights, dim=0)

        avg_hidden = torch.sum(
            x[:, self.upstream_layers_output_to_use] * w[None, :, None],
            dim=1,
        )  # (batch_size, upstream_hidden_dim)

        return avg_hidden


class FixedLayerPooling(torch.nn.Module):
    def __init__(
        self,
        *args,
        layer_idx_to_use: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.layer_idx_to_use = layer_idx_to_use

    def forward(self, x):
        """
        Get embedding from fixed layer.

        Args:
            x (torch.Tensor): Input tensor (#batch_size, #upstream_layers, upstream_hidden_dim).
        """
        return x[:, self.layer_idx_to_use]


class AttentionLayerPooling(AttentionPooling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pooled_dim = 1


class SelfAttentionLayerPooling(AttentionLayerPooling):
    def __init__(
        self,
        embed_dim: int,
        *args,
        use_positional_encoding: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(embed_dim, *args, **kwargs)
        self.attention = SelfAttentionLayer(embed_dim, *args, **kwargs)
        self.pos_encoder = (
            PositionalEncoding(embed_dim) if use_positional_encoding else None
        )
