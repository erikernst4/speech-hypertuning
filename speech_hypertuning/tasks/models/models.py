from typing import Any, Dict, List, Union

import pytorch_lightning as pl
import torch
from s3prl.nn import S3PRLUpstream


class S3PRLUpstreamMLPDownstreamForCls(pl.LightningModule):
    def __init__(
        self,
        state: Dict[str, Any],
        upstream: str = 'data2vec',
        upstream_layers_output_to_use: Union[str, List[int], int] = -1,
        hidden_layers: int = 2,
        hidden_dim: int = 128,
        optimizer_params: Dict[str, Any] = {},
    ):
        super().__init__()
        self.optimizer_params = optimizer_params
        self.mapping = state['class_map']

        self.upstream = S3PRLUpstream(upstream)
        upstream_dim = self.upstream.hidden_sizes[0]

        layer_dims = [upstream_dim] + [hidden_dim] * hidden_layers

        self.downstream = torch.nn.Sequential(
            *[
                torch.nn.Sequential(torch.nn.Linear(dim_in, dim_out), torch.nn.ReLU())
                for dim_in, dim_out in zip(layer_dims[:-1], layer_dims[1:])
            ]
        )
        self.out_layer = torch.nn.Linear(
            layer_dims[-1], len(self.mapping)
        )  # FIXME: add this at the end of the downstream?

        if isinstance(upstream_layers_output_to_use, int):
            upstream_layers_output_to_use = [upstream_layers_output_to_use]
        elif upstream_layers_output_to_use == 'all':
            upstream_layers_output_to_use = list(range(len(self.upstream.hidden_sizes)))

        self.upstream_layers_output_to_use = upstream_layers_output_to_use

        self.avg_weights = torch.nn.Parameter(
            torch.ones(
                len(upstream_layers_output_to_use),
            )
        )

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            hidden, _ = self.upstream(x['wav'], wavs_len=x['wavs_len'])
        hidden = torch.stack(hidden).transpose(0, 1)

        w = torch.nn.functional.softmax(self.avg_weights, dim=0)

        avg_hidden = torch.sum(
            hidden[:, self.upstream_layers_output_to_use] * w[None, :, None, None],
            dim=1,
        )

        return self.out_layer(self.downstream(avg_hidden))

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
    ):
        losses = self.calculate_loss(batch)
        self.log_results(losses, 'train')
        return losses

    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
    ):
        losses = self.calculate_loss(batch)
        self.log_results(losses, 'val')

    def calculate_loss(self, x: torch.Tensor):
        yhat = self(x)
        y = x['class_id']
        return torch.nn.functional.cross_entropy(yhat, y)

    def log_results(self, losses, prefix) -> None:
        self.log_dict({'{}_{}'.format(prefix, k): v for k, v in losses.items()})

    def configure_optimizers(self) -> None:
        return torch.optim.Adam(**self.optimizer_params)
