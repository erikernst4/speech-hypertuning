from typing import Optional

import torch

from .attention_poolings import (AttentionPooling, PositionalEncoding,
                                 SelfAttentionLayer, TransformerLayer)
from .summarymixing import SummaryMixing


class TemporalMeanPooling(torch.nn.Module):
    """
    Computes Temporal Mean Pooling for each layer.
    """

    def __init__(
        self,
        input_size: int,
        *args,
        before_layer_pooling: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size
        self.before_layer_pooling = (
            before_layer_pooling if before_layer_pooling is not None else True
        )
        self.time_dim = 2 if self.before_layer_pooling else 1

    def forward(self, xs: torch.Tensor, *args, **kwargs):
        """
        Compute mean along the temporal dimension

        Args:
            xs (torch.Tensor): Input tensor
                (#batch, #hidden_states, frames, hidden_dim) if before_layer_pooling,
                else (#batch, frames, hidden_dim)
        Returns:
            torch.Tensor: Output tensor
            (#batch, #hidden_states, hidden_dim) if before_layer_pooling,
            else (#batch, hidden_dim)
        """

        mean_pooled = torch.mean(xs, dim=self.time_dim)

        return mean_pooled


class AttentionTimePooling(AttentionPooling):
    def __init__(self, *args, before_layer_pooling: Optional[bool] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.before_layer_pooling = (
            before_layer_pooling if before_layer_pooling is not None else True
        )

        self.pooled_dim = 2 if self.before_layer_pooling else 1

    def prepare_attention_input(self, xs, original_shape):
        if self.before_layer_pooling:
            # x shape: (batch_size, upstream_layers, frames, embed_dim)
            batch_size, upstream_layers, frames, embed_dim = original_shape

            # Reshape to (batch_size * upstream_layers, frames, embed_dim)
            xs = xs.reshape(batch_size * upstream_layers, frames, embed_dim)
        return xs

    def post_process_attention_output(self, attn_output, original_shape):
        if self.before_layer_pooling:
            batch_size, upstream_layers, frames, embed_dim = original_shape
            # Reshape back to (batch_size, upstream_layers, frames, embed_dim)
            attn_output = attn_output.view(
                batch_size, upstream_layers, frames, embed_dim
            )
        return attn_output

    def create_padding_masks(
        self, xs: torch.Tensor, xs_len: torch.LongTensor, original_shape: torch.Size
    ) -> Optional[torch.Tensor]:
        if self.before_layer_pooling:
            batch_size, upstream_layers, frames, _ = original_shape
        else:
            batch_size, frames, _ = original_shape

        max_len = frames

        if batch_size == 1 or (xs_len == max_len).all():
            return None

        # Create the attention mask based on xs_len
        mask_base = torch.arange(max_len).expand(batch_size, max_len).to("cuda")
        mask = mask_base >= xs_len.unsqueeze(1)  # (batch_size, max_len)

        if self.before_layer_pooling:
            # Expand the mask to match the shape (batch_size * upstream_layers, max_len)
            mask = (
                mask.unsqueeze(1)
                .expand(batch_size, upstream_layers, max_len)
                .reshape(batch_size * upstream_layers, max_len)
            )
        return mask


class SelfAttentionPooling(AttentionTimePooling):
    def __init__(
        self,
        input_size: int,
        *args,
        use_positional_encoding: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(input_size, *args, **kwargs)
        self.attention = SelfAttentionLayer(input_size, *args, **kwargs)
        self.pos_encoder = (
            PositionalEncoding(input_size) if use_positional_encoding else None
        )


class TransformerPooling(AttentionTimePooling):
    def __init__(self, input_size: int, *args, **kwargs):
        super().__init__(input_size, *args, **kwargs)
        self.attention = TransformerLayer(input_size, *args, **kwargs)


class SummaryMixingPooling(AttentionTimePooling):
    def __init__(
        self, input_size: int, *args, num_heads: Optional[int] = None, **kwargs
    ):
        super().__init__(input_size, *args, **kwargs)
        num_heads = num_heads if num_heads is not None else 8
        self.attention = SummaryMixing(
            enc_dim=input_size,
            summary_out_dim=input_size,
            nhead=num_heads,
            *args,
            **kwargs,
        )
