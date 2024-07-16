import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

warnings.simplefilter("ignore", UserWarning)

import torch
import torchmetrics
from lightning import LightningModule
from loguru import logger
from s3prl.nn import S3PRLUpstream
from transformers import AutoFeatureExtractor, WavLMModel

from speech_hypertuning.tasks.models.poolings import (
    TemporalMeanPooling, WeightedAverageLayerPooling)


class PoolingProjector(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = embed_dim

        self.projector = torch.nn.Linear(embed_dim, output_dim)

    def forward(self, x: torch.Tensor):

        return self.projector(x)


class DownstreamForCls(torch.nn.Module):
    def __init__(
        self,
        state: Dict[str, Any],
        upstream_dim: int,
        hidden_layers: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ):
        if hidden_dim is None:
            logger.info("No hidden dim set for Downstream, setting to 128 as default")
            hidden_dim = 128

        if hidden_layers is None:
            logger.info("No hidden layers set for Downstream, setting to 2 as default")
            hidden_layers = 2

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
        time_pooling_layer: Optional[torch.nn.Module] = None,
        skip_pooling: Optional[bool] = None,
        frozen_upstream: Optional[bool] = None,
        normalize_upstream_embeddings: Optional[bool] = None,
        normalization_method: Optional[str] = None,
        layer_pooling_layer: Optional[torch.nn.Module] = None,
        time_pooling_before_layer_pooling: Optional[bool] = None,
        pooling_projector: Optional[bool] = None,
        downstream_cls: Optional[torch.nn.Module] = None,
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

        # Normalization variables
        self.normalize_upstream_embeddings = (
            normalize_upstream_embeddings
            if normalize_upstream_embeddings is not None
            else False
        )
        self.normalization_method = normalization_method
        self.register_buffer(
            "dataset_mean", state.get('dataset_mean', torch.tensor([]))
        )
        self.register_buffer("dataset_std", state.get('dataset_std', torch.tensor([])))

        if self.frozen_upstream:
            self.upstream.eval()

        upstream_dim = self.upstream.hidden_sizes[0]

        self.time_pooling = (
            time_pooling_layer(
                input_size=upstream_dim,
                before_layer_pooling=time_pooling_before_layer_pooling,
            )
            if time_pooling_layer is not None
            else TemporalMeanPooling(upstream_dim)
        )
        self.skip_pooling = skip_pooling if skip_pooling is not None else False

        if downstream_cls is None:
            downstream_cls = DownstreamForCls

        self.downstream = downstream_cls(
            state=state, upstream_dim=self.time_pooling.output_size
        )

        if isinstance(upstream_layers_output_to_use, int):
            upstream_layers_output_to_use = [upstream_layers_output_to_use]
        elif upstream_layers_output_to_use == 'all':
            upstream_layers_output_to_use = list(range(len(self.upstream.hidden_sizes)))

        self.upstream_layers_output_to_use = upstream_layers_output_to_use

        self.layer_pooling = (
            layer_pooling_layer(
                upstream_layers_output_to_use=self.upstream_layers_output_to_use,
                embed_dim=upstream_dim,
                before_time_pooling=not (time_pooling_before_layer_pooling),
            )
            if layer_pooling_layer is not None
            else WeightedAverageLayerPooling(self.upstream_layers_output_to_use)
        )

        self.time_pooling_before_layer_pooling = (
            time_pooling_before_layer_pooling
            if time_pooling_before_layer_pooling is not None
            else True
        )

        self.pooling_projector = (
            PoolingProjector(upstream_dim) if pooling_projector else torch.nn.Identity()
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
        upstream_embedding = self.extract_upstream_embedding(x)

        return self.downstream(upstream_embedding)

    def extract_upstream_embedding(self, x: Dict[str, Any]):
        # Forward upstream
        hidden, hidden_lens = self.forward_upstream(
            x
        )  # (batch_size, upstream_layer, frames, upstream_hidden_dim)

        # Use time pooled embeddings already calculated
        if self.skip_pooling:
            time_pooled_hidden = hidden
            time_pooled_hidden = self.normalize_features(time_pooled_hidden)
            return self.layer_pooling(time_pooled_hidden)

        normalized_hidden = self.normalize_features(hidden)

        if self.time_pooling_before_layer_pooling:
            time_pooled_hidden = self.time_pooling(normalized_hidden, hidden_lens)
            time_pooled_hidden = self.pooling_projector(time_pooled_hidden)
            upstream_embedding = self.layer_pooling(time_pooled_hidden)
        else:
            layer_pooled_hidden = self.layer_pooling(normalized_hidden)
            layer_pooled_hidden = self.pooling_projector(layer_pooled_hidden)
            upstream_embedding = self.time_pooling(layer_pooled_hidden, hidden_lens)

        return upstream_embedding

    def forward_upstream(self, x: Dict[str, Any]) -> torch.Tensor:
        if (
            not "upstream_embedding_precalculated" in x
            or not x["upstream_embedding_precalculated"].all().item()
        ):  # Check if all instances have the embedding precalculated
            with torch.no_grad():
                hidden, hidden_lens = self.upstream(x['wav'], wavs_len=x['wav_lens'])

            # Out to tensor
            hidden = torch.stack(hidden).transpose(0, 1)
            hidden_lens = hidden_lens[
                0
            ]  # Before it's a list of length=upstream_layers, all elements are equal
        else:
            hidden = x['upstream_embedding']
            hidden_lens = None

        return hidden, hidden_lens

    def normalize_features(self, xs: torch.Tensor):
        if self.normalize_upstream_embeddings:
            if self.normalization_method == "standard_normalization":
                xs = torch.nn.functional.normalize(xs, dim=-1)
            elif self.normalization_method == "dataset_scaling":
                xs -= self.dataset_mean
                xs /= self.dataset_std
        return xs

    def training_step(  # pylint: disable=arguments-differ
        self, batch: torch.Tensor
    ) -> torch.Tensor:
        losses = self.calculate_loss(batch)
        normalized_loss = self.calculate_normalized_loss(losses)

        self.log_results(losses, 'train')
        self.log_results(normalized_loss, 'train', 'normalized_loss')
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


class HFUpstreamMLPDownstreamForCls(LightningModule):
    def __init__(
        self,
        state: Dict[str, Any],
        upstream: str = 'microsoft/wavlm-base-plus',
        upstream_layers_output_to_use: Union[str, List[int], int] = 'all',
        optimizer: Optional[Any] = None,
        lr_scheduler: Optional[Any] = None,
        pooling_layer: Optional[torch.nn.Module] = None,
        skip_pooling: Optional[bool] = None,
        frozen_upstream: Optional[bool] = None,
        normalize_upstream_embeddings: Optional[bool] = None,
        normalization_method: Optional[str] = None,
    ):
        super().__init__()
        self.opt_state = state
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam
        self.lr_scheduler = lr_scheduler
        self.mapping = state['speaker_id_mapping']
        self.num_classes = len(self.mapping)

        # Precalculating to calculate normalized loss
        self.prior_distribution_entropy = state["prior_distribution_entropy"]

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            upstream, return_attention_mask=True
        )
        self.upstream = WavLMModel.from_pretrained(upstream)
        self.frozen_upstream = frozen_upstream if frozen_upstream is not None else False

        # Normalization variables
        self.normalize_upstream_embeddings = (
            normalize_upstream_embeddings
            if normalize_upstream_embeddings is not None
            else False
        )
        self.normalization_method = normalization_method
        self.register_buffer(
            "dataset_mean", state.get('dataset_mean', torch.tensor([]))
        )
        self.register_buffer("dataset_std", state.get('dataset_std', torch.tensor([])))

        if self.frozen_upstream:
            self.upstream.eval()

        upstream_dim = 768
        upstream_layers = 13

        self.pooling = (
            pooling_layer(upstream_dim)
            if pooling_layer is not None
            else TemporalMeanPooling(upstream_dim)
        )
        self.skip_pooling = skip_pooling if skip_pooling is not None else False

        self.downstream = DownstreamForCls(
            state=state, upstream_dim=self.pooling.output_size
        )

        if isinstance(upstream_layers_output_to_use, int):
            upstream_layers_output_to_use = [upstream_layers_output_to_use]
        elif upstream_layers_output_to_use == 'all':
            upstream_layers_output_to_use = list(range(upstream_layers))

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
        upstream_embedding = self.extract_upstream_embedding(x)

        return self.downstream(upstream_embedding)

    def extract_upstream_embedding(self, x: Dict[str, Any]):
        # Forward upstream
        hidden, hidden_lens = self.forward_upstream(
            x
        )  # (batch_size, upstream_layer, frames, upstream_hidden_dim)

        # Summarize frames dimension if necessary (batch_size, upstream_layer, upstream_hidden_dim)
        if self.skip_pooling:
            pooled_hidden = hidden
            pooled_hidden = self.normalize_features(pooled_hidden)
        else:
            normalized_hidden = self.normalize_features(hidden)
            pooled_hidden = self.pooling(normalized_hidden, hidden_lens)

        # Summarize layers embeddings
        w = torch.nn.functional.softmax(self.avg_weights, dim=0)

        avg_hidden = torch.sum(
            pooled_hidden[:, self.upstream_layers_output_to_use] * w[None, :, None],
            dim=1,
        )  # (batch_size, upstream_hidden_dim)

        return avg_hidden

    def forward_upstream(self, x: Dict[str, Any]) -> torch.Tensor:
        if (
            not "upstream_embedding_precalculated" in x
            or not x["upstream_embedding_precalculated"].all().item()
        ):  # Check if all instances have the embedding precalculated
            with torch.no_grad():
                out = self.upstream(x['wav'], output_hidden_states=True)
                hidden = torch.stack(out['hidden_states']).transpose(0, 1)

            # Out to tensor
            hidden_lens = torch.ones(hidden.size(0), dtype=torch.int) * hidden.size(2)
        else:
            hidden = x['upstream_embedding']
            hidden_lens = None

        return hidden, hidden_lens

    def normalize_features(self, xs: torch.Tensor):
        if self.normalize_upstream_embeddings:
            if self.normalization_method == "standard_normalization":
                xs = torch.nn.functional.normalize(xs, dim=-1)
            elif self.normalization_method == "dataset_scaling":
                xs -= self.dataset_mean
                xs /= self.dataset_std
        return xs

    def training_step(  # pylint: disable=arguments-differ
        self, batch: torch.Tensor
    ) -> torch.Tensor:
        losses = self.calculate_loss(batch)
        normalized_loss = self.calculate_normalized_loss(losses)

        self.log_results(losses, 'train')
        self.log_results(normalized_loss, 'train', 'normalized_loss')
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
