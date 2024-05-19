import torch


class TemporalMeanPooling(torch.nn.Module):
    """
    Computes Temporal Mean Pooling for each layer.
    """

    def forward(self, xs: torch.Tensor, xs_len: torch.LongTensor):
        """
        Args:
            xs (torch.Tensor): Input tensor (#batch, #hidden_states, input_size, hidden_dim).
        Returns:
            torch.Tensor: Output tensor (#batch, input_size)
        """
        pooled_list = []
        for x_idx, x in enumerate(xs):
            pooled_layers = []
            for layer_idx, layer in enumerate(x):
                cut_padding_idx = xs_len[layer_idx][x_idx]
                pooled_layer = torch.mean(layer[:cut_padding_idx], dim=0)
                pooled_layers.append(pooled_layer)
            pooled_list.append(torch.stack(pooled_layers))

        return torch.stack(pooled_list)
