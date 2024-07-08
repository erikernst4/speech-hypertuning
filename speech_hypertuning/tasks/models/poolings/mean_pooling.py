import torch


class TemporalMeanPooling(torch.nn.Module):
    """
    Computes Temporal Mean Pooling for each layer.
    """

    def __init__(self, input_size: int, *args, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size

    def forward(self, xs: torch.Tensor):
        """
        Args:
            xs (torch.Tensor): Input tensor (#batch, #hidden_states, input_size, hidden_dim).
        Returns:
            torch.Tensor: Output tensor (#batch, input_size)
        """

        # Compute mean along the temporal dimension
        mean_pooled = torch.mean(xs, dim=2)

        return mean_pooled
