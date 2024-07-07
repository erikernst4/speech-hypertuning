import math
from typing import Optional

import torch

from .summarymixing import SummaryMixing


class AttentionPooling(torch.nn.Module):
    def __init__(
        self, input_size: int, *args, dropout: Optional[float] = None, **kwargs
    ):
        super().__init__()
        self.attention = None
        self.pos_encoder = None
        self.input_size = input_size
        self.output_size = input_size

        self.dropout = torch.nn.Dropout(p=dropout) if dropout is not None else None

    def forward(self, xs: torch.Tensor, xs_len: torch.LongTensor):
        """
        Args:
            xs (torch.Tensor): Input tensor (#batch, #hidden_states, #frames, hidden_dim).
            xs_len (torch.LongTensor): with the lengths for each sample  (batch_size)
        Returns:
            torch.Tensor: Output tensor (#batch, #hidden_states, output_size)
        """
        padding_mask = self.create_padding_masks(xs, xs_len)

        # x shape: (batch_size, upstream_layers, frames, embed_dim)
        batch_size, upstream_layers, frames, embed_dim = xs.size()

        # Reshape to (batch_size * upstream_layers, frames, embed_dim)
        xs = xs.reshape(batch_size * upstream_layers, frames, embed_dim)

        if self.pos_encoder is not None:
            xs = self.pos_encoder(xs)

        if self.dropout is not None:
            xs = self.dropout(xs)

        attn_output = self.attention(
            xs, padding_mask
        )  # (batch_size, upstream_layers, frames, upstream_hidden_dim)

        # Reshape back to (batch_size, upstream_layers, frames, embed_dim)
        attn_output = attn_output.view(batch_size, upstream_layers, frames, embed_dim)

        pooled_output = torch.mean(
            attn_output, dim=2
        )  # (batch_size, upstream_layers, upstream_hidden_dim)

        return pooled_output

    def create_padding_masks(
        self, xs: torch.Tensor, xs_len: torch.LongTensor
    ) -> Optional[torch.Tensor]:
        batch_size, upstream_layers, frames, embed_dim = xs.size()

        max_len = frames

        if batch_size == 1 or (xs_len == max_len).all():
            return None

        # Create the attention mask based on xs_len
        mask_base = torch.arange(max_len).expand(batch_size, max_len).to("cuda")
        mask = mask_base >= xs_len.unsqueeze(1)

        # Expand the mask to match the shape (batch_size * upstream_layers, max_len)
        mask = (
            mask.unsqueeze(1)
            .expand(batch_size, upstream_layers, max_len)
            .reshape(batch_size * upstream_layers, max_len)
        )
        return mask


class SummaryMixingPooling(AttentionPooling):
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


class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
        dim_feedforward: Optional[int] = None,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        num_heads = num_heads if num_heads is not None else 1
        num_layers = num_layers if num_layers is not None else 1
        dim_feedforward = dim_feedforward if dim_feedforward is not None else 2048
        dropout = dropout if dropout is not None else 0.1

        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

    def forward(
        self,
        x,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            xs (torch.Tensor): Input tensor (#batch, #hidden_states, #frames, hidden_dim).
            xs_len (torch.LongTensor): with the lengths for each sample (#hidden_states, batch_size)
        Returns:
            torch.Tensor: Output tensor (#batch, output_size)
        """
        # Apply transformer encoder with the key padding mask
        attn_output = self.transformer_encoder(x, src_key_padding_mask=mask)

        return attn_output


class TransformerPooling(AttentionPooling):
    def __init__(self, input_size: int, *args, **kwargs):
        super().__init__(input_size, *args, **kwargs)
        self.attention = TransformerLayer(input_size, *args, **kwargs)


class SelfAttentionLayer(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *args,
        num_heads: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        num_heads = num_heads if num_heads is not None else 1
        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

    def forward(
        self,
        x,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            xs (torch.Tensor): Input tensor (#batch, #hidden_states, #frames, hidden_dim).
            xs_len (torch.LongTensor): with the lengths for each sample (#hidden_states, batch_size)
        Returns:
            torch.Tensor: Output tensor (#batch, output_size)
        """
        # Apply multihead attention
        attn_output, _ = self.multihead_attn(
            x, x, x, key_padding_mask=mask, need_weights=False
        )

        return attn_output


class PositionalEncoding(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        max_len: int = 5000,
    ):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, xs):
        """
        Adds positional encoding to the input.

        Args:
            xs (torch.Tensor): Input tensor (#batch * #hidden_states, #frames, hidden_dim).
        Returns:
            torch.Tensor: Output tensor #batch * #hidden_states, #frames, hidden_dim)
        """
        seq_len = xs.size(1)
        return xs + self.pe[:, :seq_len]


class SelfAttentionPooling(AttentionPooling):
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
