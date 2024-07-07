from typing import List

import torch


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
