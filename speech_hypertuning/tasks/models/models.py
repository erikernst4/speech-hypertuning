from typing import Any
import pytorch_lightning as pl
from torch import nn
from s3prl.nn import S3PRLUpstream


class WavLMBasePlusForSpeakerIdentification(pl.LightningModule):
    def __init__(
        self,
        downstream_size: int,
    ) -> None:
        super().__init__()
        self.upstream = S3PRLUpstream("wavlm_base_plus")
        self.downstream = nn.Linear(downstream_size)

    def forward(self, x):
        return self.downstream(self.upstream(x))