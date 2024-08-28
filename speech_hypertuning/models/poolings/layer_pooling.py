from typing import List, Optional

import torch

from .attention_poolings import (AttentionPooling, PositionalEncoding,
                                 SelfAttentionLayer, TransformerLayer)


class WeightedAverageLayerPooling(torch.nn.Module):
    def __init__(
        self,
        upstream_layers_output_to_use: List[int],
        *args,
        before_time_pooling: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__()
        self.upstream_layers_output_to_use = upstream_layers_output_to_use

        self.avg_weights = torch.nn.Parameter(
            torch.ones(
                len(upstream_layers_output_to_use),
            )
        )

        self.before_time_pooling = (
            before_time_pooling if before_time_pooling is not None else False
        )

    def forward(self, x):
        """
        Calculate weighted average of the layers embeddings.

        Args:
            x (torch.Tensor): Input tensor
                (#batch, #hidden_states, frames, hidden_dim) if before time pooling,
                else (#batch, #hidden_states, hidden_dim)
        Returns:
            torch.Tensor: Output tensor
            (#batch, frames, hidden_dim) if before time pooling,
            else (#batch, hidden_dim)
        """
        # x shape (batch_size, upstream_layer, upstream_hidden_dim)
        w = torch.nn.functional.softmax(self.avg_weights, dim=0)

        if self.before_time_pooling:
            layer_weighted_x = (
                x[:, self.upstream_layers_output_to_use] * w[None, :, None, None]
            )
        else:
            layer_weighted_x = (
                x[:, self.upstream_layers_output_to_use] * w[None, :, None]
            )

        avg_hidden = torch.sum(
            layer_weighted_x,
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
            x (torch.Tensor): Input tensor
                (#batch, #hidden_states, frames, hidden_dim) if before time pooling,
                else (#batch, #hidden_states, hidden_dim)
        Returns:
            torch.Tensor: Output tensor
            (#batch, frames, hidden_dim) if before time pooling,
            else (#batch, hidden_dim)
        """
        return x[:, self.layer_idx_to_use]


class AttentionLayerPooling(AttentionPooling):
    def __init__(self, *args, before_time_pooling: Optional[bool] = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.pooled_dim = 1

        self.before_time_pooling = (
            before_time_pooling if before_time_pooling is not None else False
        )

    def prepare_attention_input(self, xs, original_shape):
        if self.before_time_pooling:
            # x shape: (batch_size, upstream_layers, frames, embed_dim)
            batch_size, upstream_layers, frames, embed_dim = original_shape

            # Permute to keep the dimensions meaning
            xs = torch.permute(xs, (0, 2, 1, 3))

            # Reshape to (N, |seq|, embed_dim)
            xs = xs.reshape(batch_size * frames, upstream_layers, embed_dim)
        return xs

    def post_process_attention_output(self, attn_output, original_shape):
        if self.before_time_pooling:
            batch_size, upstream_layers, frames, embed_dim = original_shape

            # Reshape back to (batch_size, upstream_layers, frames, embed_dim)
            attn_output = attn_output.view(
                batch_size, frames, upstream_layers, embed_dim
            )

            attn_output = torch.permute(attn_output, (0, 2, 1, 3))

        return attn_output


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

class TransformerLayerPooling(AttentionLayerPooling):
    def __init__(self, embed_dim: int, *args, **kwargs):
        super().__init__(embed_dim, *args, **kwargs)
        self.attention = TransformerLayer(embed_dim, *args, **kwargs)