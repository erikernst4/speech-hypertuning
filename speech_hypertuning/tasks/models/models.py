from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import warnings
warnings.simplefilter("ignore", UserWarning)

import numpy as np
import torch
import torchmetrics
from lightning import LightningModule
from s3prl.nn import S3PRLUpstream

from speech_hypertuning.tasks.models.pooling import TemporalMeanPooling


class DownstreamForCls(torch.nn.Module):
    def __init__(
        self,
        state: Dict[str, Any],
        upstream_dim: int,
        hidden_layers: int = 2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.opt_state = state
        self.mapping = state['speaker_id_mapping']
        self.num_classes = len(self.mapping)

        layer_dims = [upstream_dim] + [hidden_dim] * hidden_layers

        self.hidden_net = torch.nn.Sequential(
            *[
                torch.nn.Sequential(torch.nn.Linear(dim_in, dim_out), torch.nn.ReLU())
                for dim_in, dim_out in zip(layer_dims[:-1], layer_dims[1:])
            ]
        )
        self.out_layer = torch.nn.Linear(layer_dims[-1], self.num_classes)

    def forward(self, upstream_avg_hidden: torch.Tensor):

        return self.out_layer(self.hidden_net(upstream_avg_hidden))


class S3PRLUpstreamMLPDownstreamForCls(LightningModule):
    def __init__(
        self,
        state: Dict[str, Any],
        upstream: str = 'wavlm_base_plus',
        upstream_layers_output_to_use: Union[str, List[int], int] = 'all',
        optimizer: Optional[Any] = None,
        lr_scheduler: Optional[Any] = None,
        pooling_layer: Optional[torch.nn.Module] = None,
        frozen_upstream: Optional[bool] = None,
        normalize_upstream_embeddings: Optional[bool] = None,
    ):
        super().__init__()
        self.opt_state = state
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam
        self.lr_scheduler = lr_scheduler
        self.mapping = state['speaker_id_mapping']
        self.num_classes = len(self.mapping)

        # Precalculating to calculate normalized loss
        self.prior_distribution_entropy = state["prior_distribution_entropy"]

        self.upstream = S3PRLUpstream(upstream)
        self.frozen_upstream = frozen_upstream if frozen_upstream is not None else False
        self.normalize_upstream_embeddings = normalize_upstream_embeddings if normalize_upstream_embeddings is not None else False

        if self.frozen_upstream:
            self.upstream.eval()

        upstream_dim = self.upstream.hidden_sizes[0]

        self.pooling = (
            pooling_layer if pooling_layer is not None else TemporalMeanPooling()
        )

        self.downstream = DownstreamForCls(state=state, upstream_dim=upstream_dim)

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

        self.accuracy_top1 = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.accuracy_top5 = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=self.num_classes,
            top_k=min(5, self.num_classes),
        )

    def forward(self, x: Dict[str, Any]):  # pylint: disable=arguments-differ

        hidden = self.forward_upstream(x)

        pooled_hidden = self.pooling(hidden)

        w = torch.nn.functional.softmax(self.avg_weights, dim=0)

        avg_hidden = torch.sum(
            pooled_hidden[:, self.upstream_layers_output_to_use] * w[None, :, None],
            dim=1,
        )

        return self.downstream(avg_hidden)

    def forward_upstream(self, x: Dict[str, Any]) -> torch.Tensor:
        if (
            not "upstream_embedding_precalculated" in x
            or not x["upstream_embedding_precalculated"].all().item()
        ):  # Check if all instances have the embedding precalculated
            with torch.no_grad():
                hidden, _ = self.upstream(x['wav'], wavs_len=x['wav_lens'])

            # Out to tensor
            hidden = torch.stack(hidden).transpose(0, 1)
        else:
            hidden = x['upstream_embedding']

            # Add batch size dimension if necessary
            if len(hidden.shape) == 3 and hidden.size(0) > 1:
                hidden = hidden.unsqueeze(dim=0)

        if self.normalize_upstream_embeddings:
            hidden = torch.nn.functional.normalize(hidden, dim=3)

        return hidden

    def training_step(  # pylint: disable=arguments-differ
        self, batch: torch.Tensor
    ) -> torch.Tensor:
        losses = self.calculate_loss(batch)
        self.log_results(losses, 'train')
        return losses

    def validation_step(  # pylint: disable=arguments-differ
        self, batch: torch.Tensor
    ) -> None:
        losses = self.calculate_loss(batch)
        normalized_loss = self.calculate_normalized_loss(losses)

        self.log_results(losses, 'val')
        self.log_results(normalized_loss, 'val', 'normalized_loss')

    def test_step(  # pylint: disable=arguments-differ
        self, batch: torch.Tensor
    ) -> None:
        losses = self.calculate_loss(batch)

        out = self(batch)
        yhat = out.squeeze()
        if len(yhat.shape) == 1:
            yhat = yhat.unsqueeze(dim=0)
        y = batch['class_id']

        accuracy_top1 = self.accuracy_top1(yhat, y)
        accuracy_top5 = self.accuracy_top5(yhat, y)
        normalized_loss = self.calculate_normalized_loss(losses)

        self.log_results(losses, 'test')
        self.log_results(accuracy_top1, 'test', 'accuracy_top1')
        self.log_results(accuracy_top5, 'test', 'accuracy_top5')
        self.log_results(normalized_loss, 'test', 'normalized_loss')

    def calculate_loss(self, x: torch.Tensor) -> torch.Tensor:
        out = self(x)
        yhat = out.squeeze()
        y = x['class_id']

        if len(yhat.shape) == 1:
            yhat = yhat.unsqueeze(dim=0)

        return torch.nn.functional.cross_entropy(yhat, y)

    def calculate_normalized_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Dividing the loss by the entropy of the prior distribution
        gives us an interpretable metric for which a value larger than 1.0
        indicates that the system is worse than a naive system.

        For more information check:
            - Ferrer, L. (2022). Analysis and Comparison of Classification Metrics. ArXiv, abs/2209.05355.
        """
        return loss / self.prior_distribution_entropy

    def log_results(self, losses: torch.Tensor, prefix, metric="loss") -> None:
        log_loss = {
            "time": int(datetime.now().strftime('%y%m%d%H%M%S')),
            metric: losses,
        }
        self.log_dict({'{}_{}'.format(prefix, k): v for k, v in log_loss.items()})

    def configure_optimizers(
        self,
    ) -> Dict[str, Any]:
        optimizer = self.optimizer(params=self.parameters())
        optimizer_config = {"optimizer": optimizer}
        if self.lr_scheduler is not None:
            lr_scheduler_config = {
                "scheduler": self.lr_scheduler(optimizer),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
            optimizer_config['lr_scheduler'] = lr_scheduler_config
        return optimizer_config

    def set_optimizer_state(self, state: Dict[str, Any]) -> None:
        self.opt_state = state
