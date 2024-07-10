from typing import List, Optional

import torch

from speech_hypertuning.tasks.models.poolings.attention_poolings import (
    PositionalEncoding, SelfAttentionLayer)


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


class AttentionLayerPooling(torch.nn.Module):
    def __init__(
        self, input_size: int, *args, dropout: Optional[float] = None, **kwargs
    ):
        super().__init__()
        self.attention = None
        self.pos_encoder = None
        self.input_size = input_size
        self.output_size = input_size

        self.dropout = torch.nn.Dropout(p=dropout) if dropout is not None else None

    def forward(self, xs: torch.Tensor):
        """
        Args:
            xs (torch.Tensor): Input tensor (#batch, #hidden_states, hidden_dim).
            xs_len (torch.LongTensor): with the lengths for each sample  (batch_size)
        Returns:
            torch.Tensor: Output tensor (#batch, #hidden_states, output_size)
        """

        if self.pos_encoder is not None:
            xs = self.pos_encoder(xs)

        if self.dropout is not None:
            xs = self.dropout(xs)

        attn_output = self.attention(
            xs
        )  # (batch_size, upstream_layers, upstream_hidden_dim)

        pooled_output = torch.mean(
            attn_output, dim=1
        )  # (batch_size, upstream_hidden_dim)

        return pooled_output


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
