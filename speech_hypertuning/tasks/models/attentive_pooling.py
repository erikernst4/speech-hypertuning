from typing import Optional, Tuple

import torch


class SelfAttentionLayer(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: Optional[int] = None,
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xs (torch.Tensor): Input tensor (#batch, #hidden_states, #frames, hidden_dim).
            mask (torch.BoolTensor): Padding mask for the input tensor (#batch, #hidden_states, #frames)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output attention tensor and weights ((#batch, #hidden_states, #frames, hidden_dim), (#batch, #hidden_states, #frames, #frames))
        """
        # x shape: (batch_size, upstream_layers, frames, embed_dim)
        batch_size, upstream_layers, frames, embed_dim = x.size()

        # Reshape to (batch_size * upstream_layers, frames, embed_dim)
        x = x.reshape(batch_size * upstream_layers, frames, embed_dim)

        # Apply multihead attention
        attn_output, attn_output_weights = self.multihead_attn(
            x, x, x, key_padding_mask=mask
        )

        # Reshape back to (batch_size, upstream_layers, frames, embed_dim)
        attn_output = attn_output.view(batch_size, upstream_layers, frames, embed_dim)
        attn_output_weights = attn_output_weights.view(
            batch_size, upstream_layers, frames, frames
        )

        return attn_output, attn_output_weights


class SelfAttentionPooling(torch.nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.attention = SelfAttentionLayer(input_size)
        self.input_size = input_size
        self.output_size = input_size

    def forward(self, xs: torch.Tensor, xs_len: torch.LongTensor):
        """
        Args:
            xs (torch.Tensor): Input tensor (#batch, #hidden_states, #frames, hidden_dim).
            xs_len (torch.LongTensor): with the lengths for each sample  (batch_size)
        Returns:
            torch.Tensor: Output tensor (#batch, #hidden_states, output_size)
        """
        padding_mask = self.create_padding_masks(xs, xs_len)

        attn_output, attn_output_weights = self.attention(xs, padding_mask)

        # Compute the average attention weights across the frames dimension
        avg_attn_weights = attn_output_weights.mean(
            dim=2
        )  # (batch_size, upstream_layers, frames)

        # Compute weighted sum of the frame features
        pooled_output = torch.sum(attn_output * avg_attn_weights.unsqueeze(-1), dim=2)

        return pooled_output

    def create_padding_masks(self, xs: torch.Tensor, xs_len: torch.LongTensor):
        batch_size, upstream_layers, frames, embed_dim = xs.size()
        max_len = frames

        # Create the attention mask based on xs_len
        mask_base = torch.arange(max_len).expand(batch_size, max_len).to('cuda')
        mask = mask_base >= xs_len.unsqueeze(1)

        # Expand the mask to match the shape (batch_size * upstream_layers, max_len)
        mask = (
            mask.unsqueeze(1)
            .expand(batch_size, upstream_layers, max_len)
            .reshape(batch_size * upstream_layers, max_len)
        )
        return mask
