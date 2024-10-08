import math
from typing import Optional

import torch


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

    def forward(self, xs: torch.Tensor, xs_len: Optional[torch.LongTensor] = None):
        """
        Args:
            xs (torch.Tensor): Input tensor.
            xs_len (torch.LongTensor): with the lengths for each sample  (batch_size)
        Returns:
            torch.Tensor: Output tensor (#batch, #hidden_states, output_size)
        """
        if self.pooled_dim is None:
            raise ValueError("Dimension to pool not set.")

        original_shape = xs.size()

        padding_mask = self.create_padding_masks(xs, xs_len, original_shape)

        xs = self.prepare_attention_input(xs, original_shape)

        # (N, |seq|, embed_dim)
        if self.pos_encoder is not None:
            xs = self.pos_encoder(xs)

        if self.dropout is not None:
            xs = self.dropout(xs)

        attn_output = self.attention(xs, padding_mask)

        attn_output = self.post_process_attention_output(attn_output, original_shape)

        pooled_output = torch.mean(attn_output, dim=self.pooled_dim)

        return pooled_output

    def prepare_attention_input(self, xs, original_shape):
        return xs

    def post_process_attention_output(self, attn_output, original_shape):
        return attn_output

    def create_padding_masks(
        self, xs: torch.Tensor, xs_len: torch.LongTensor, original_shape: torch.Size
    ) -> Optional[torch.Tensor]:
        return None


class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *args,
        num_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
        dim_feedforward: Optional[int] = None,
        dropout: Optional[float] = None,
        **kwargs
    ):
        super().__init__()
        num_heads = num_heads if num_heads is not None else 16
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


class SelfAttentionLayer(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *args,
        num_heads: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        num_heads = num_heads if num_heads is not None else 16
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
            xs (torch.Tensor): Input tensor (N, #frames, hidden_dim).
        Returns:
            torch.Tensor: Output tensor N, #frames, hidden_dim)
        """
        seq_len = xs.size(1)
        return xs + self.pe[:, :seq_len]
