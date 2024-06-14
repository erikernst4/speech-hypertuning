""" SummaryMixing © 2023 by Samsung Electronics is licensed under CC BY-NC 4.0.

This library provides the basic building blocks for SummaryMixing.

Usage: Install SpeechBrain and copy this file under speechbrain/nnet/
Source: https://arxiv.org/abs/2307.07421

Authors
 * Titouan Parcollet 2023
 * Shucong Zhang 2023
 * Rogier van Dalen 2023
 * Sourav Bhattacharya 2023
"""

import logging
import math
from typing import Optional

import speechbrain as sb
import torch
import torch.nn as nn
from speechbrain.nnet.containers import Sequential


class SummaryMixing(nn.Module):
    """This class implements SummaryMixing as defined
    in https://arxiv.org/abs/2307.07421

    Arguments
    ---------
    enc_dim: int
        Feature dimension of the input tensor.
    nhead : int
        Number of mixing heads.
    local_proj_hid_dim: list [int], optional
        A list of dimension specifying both the number of hidden layers
        as well as the size of them in the local projection branch
        (default: [512]).
    local_proj_out_dim: int, optional
        The dimension of the output of the local projection branch. This
        will be concatenated with the output of the summary branch
        (default: 512).
    summary_hid_dim: list [int], optional
        A list of dimension specifying both the number of hidden layers
        as well as the size of them in the summary projection branch
        (default: [512]).
    summary_out_dim: int, optional
        The dimension of the output of the summary projection branch. This
        will be concatenated with the output of the local branch
        (default: 512).
    activation: torch.nn.Module, optional
        Torch module specifying the activation function used in both the local
        and summary branches.
        (default: torch.nn.GELU)
    mode: string, optional
        One of "SummaryMixing" or "SummaryMixing-lite". Changes the SummaryMixing cell
        according to the definition of the article. "SummaryMixing-lite" removes the
        local project branch.


    Example
    -------
    >>> x = torch.rand(2,4,8)
    >>> sum = SummaryMixing(8)
    >>> out = sum(x)
    >>> print(out)
    torch.Size([2, 4, 8])
    """

    def __init__(
        self,
        enc_dim,
        nhead,
        local_proj_hid_dim: Optional[list] = [512],
        local_proj_out_dim: Optional[int] = 512,
        summary_hid_dim: Optional[list] = [512],
        summary_out_dim: Optional[int] = 512,
        activation: Optional[nn.Module] = nn.GELU,
        mode: Optional[str] = "SummaryMixing",
    ):
        super(SummaryMixing, self).__init__()

        if mode not in ["SummaryMixing", "SummaryMixing-lite"]:
            raise ValueError(
                "The SummaryMixing mode should either be 'SummaryMixing' or 'SummaryMixing-lite'"
            )

        self.local_proj_hid_dim = local_proj_hid_dim
        self.local_proj_out_dim = local_proj_out_dim
        self.summary_hid_dim = summary_hid_dim
        self.summary_out_dim = summary_out_dim
        self.enc_dim = enc_dim
        self.activation = activation()
        self.local_dnn_blocks = local_proj_hid_dim + [local_proj_out_dim]
        self.summary_dnn_blocks = summary_hid_dim + [summary_out_dim]
        self.mode = mode

        if self.mode == "SummaryMixing":

            self.local_proj = VanillaNN(
                input_shape=[None, None, enc_dim],
                dnn_blocks=len(self.local_dnn_blocks),
                dnn_neurons=self.local_dnn_blocks,
                activation=activation,
                n_split=nhead,
            )

            self.summary_local_merging = VanillaNN(
                input_shape=[None, None, local_proj_out_dim + summary_out_dim],
                dnn_blocks=1,
                dnn_neurons=[summary_out_dim],
                activation=activation,
            )

            self.local_norm = nn.LayerNorm(local_proj_out_dim)
            self.summary_norm = nn.LayerNorm(summary_out_dim)

        self.summary_proj = VanillaNN(
            input_shape=[None, None, enc_dim],
            dnn_blocks=len(self.summary_dnn_blocks),
            dnn_neurons=self.summary_dnn_blocks,
            activation=activation,
            n_split=nhead,
        )

        self.apply(self._init_parameters)

    def forward(self, x, attention_mask=None):
        """This function simply goes forward!

        Arguments
        ---------
        x: torch.Tensor
            The expected shape is the standard SpeechBrain one - [Batch, Time, Features]
        attention_mask: torch.Tensor
            (B, S) to pad before summarizing in time.
        """

        if attention_mask is not None:
            attention_mask = torch.logical_not(attention_mask).unsqueeze(-1).float()
        else:
            attention_mask = torch.ones((x.shape[0], x.shape[1])).unsqueeze(-1).float()

        if self.mode == "SummaryMixing":
            return self._forward_mixing(x, attention_mask)
        elif self.mode == "SummaryMixing-lite":
            return self._forward_avgonly(x, attention_mask)

    def _forward_mixing(self, x, attention_mask):
        """Perform full SummaryMixing.

        Arguments
        ---------
        x: torch.Tensor
            The expected shape is the standard SpeechBrain one - [Batch, Time, Features]
        attention_mask: torch.Tensor
            (B, S) to pad before summarizing in time.
        """

        B, T, F = x.shape

        # f() (Eq. 1b)
        local_summary = self.local_norm(self.local_proj(x) * attention_mask)

        # s() (Eq. 2 and 1c)
        time_summary = self.summary_proj(x) * attention_mask

        # We normalise by real length by counting masking
        time_summary = self.summary_norm(
            torch.sum(time_summary, dim=1) / torch.sum(attention_mask, dim=1)
        )
        time_summary = time_summary.unsqueeze(1).repeat(1, T, 1)

        return self.summary_local_merging(
            torch.cat([local_summary, time_summary], dim=-1)
        )

    def _forward_avgonly(self, x, attention_mask):
        """Perform SummaryMixing-lite.

        Arguments
        ---------
        x: torch.Tensor
            The expected shape is the standard SpeechBrain one - [Batch, Time, Features]
        attention_mask: torch.Tensor
            (B, S) to pad before summarizing in time.
        """

        B, T, F = x.shape

        # s() We just do the mean over time
        # Then we repeat the output matrix T times along the time axis
        time_summary = self.summary_proj(x) * attention_mask
        time_summary = torch.sum(time_summary, dim=1) / torch.sum(attention_mask, dim=1)
        time_summary = time_summary.unsqueeze(1).expand(-1, T, -1)

        return time_summary

    def _init_parameters(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.zeros_(module.bias)

    def _reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.A_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B_weights, a=math.sqrt(5))


logger = logging.getLogger(__name__)


class ParallelLinear(torch.nn.Module):
    """Computes a parallel linear transformation y = wx + b.
    In practice the input and the output are split n_split times.
    Hence we create n_split parallel linear op that will operate on
    each splited dimension. E.g. if x = [B,T,F] and n_split = 4
    then x = [B,T,4,F/4] and W = [4,F/4,out_dim/4].

    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_shape: tuple, optional
        It is the shape of the input tensor.
    input_size: int, optional
        Size of the input tensor.
    n_split: int, optional
        The number of split to create n_split linear transformations.
    bias : bool, optional
        If True, the additive bias b is adopted.
    combiner_out_dims : bool, optional
        If True, the output vector is reshaped to be [B, T, S].

    Example
    -------
    >>> x = torch.rand([64, 50, 512])
    >>> lin_t = ParallelLinear(n_neurons=64, input_size=512, n_split=4)
    >>> output = lin_t(x)
    >>> output.shape
    torch.Size([64, 50, 64])
    """

    def __init__(
        self,
        n_neurons,
        input_shape: Optional[list] = None,
        input_size: Optional[int] = None,
        n_split: Optional[int] = 1,
        bias: Optional[bool] = True,
        combine_out_dims: Optional[bool] = True,
    ):
        super().__init__()
        self.n_split = n_split
        self.combine_out_dims = combine_out_dims

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]
            if len(input_shape) == 4:
                input_size = input_shape[-1] * input_shape[-2]

        if input_size % n_split != 0 or n_neurons % n_split != 0:
            raise ValueError("input_size and n_neurons must be dividible by n_split!")

        self.split_inp_dim = input_size // n_split
        self.split_out_dim = n_neurons // n_split

        self.weights = nn.Parameter(
            torch.empty(self.n_split, self.split_inp_dim, self.split_out_dim)
        )
        self.biases = nn.Parameter(torch.zeros(self.n_split, self.split_out_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.biases, a=math.sqrt(5))

    def forward(self, x):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input to transform linearly, may be 3 or four dimensional.
            [B,T,F] or [B,T,n_split,F//n_split]
        """
        if x.ndim == 3:
            B, T, F = x.shape
            x = x.view(B, T, self.n_split, self.split_inp_dim)

        x = torch.einsum("btmf,mfh->btmh", x, self.weights) + self.biases

        if self.combine_out_dims:
            x = x.reshape(x.shape[0], x.shape[1], -1)

        return x


class VanillaNN(Sequential):
    """A simple vanilla Deep Neural Network.

    Arguments
    ---------
    activation : torch class
        A class used for constructing the activation layers.
    dnn_blocks : int
        The number of linear neural blocks to include.
    dnn_neurons : int or list[int]
        The number of neurons in the different linear layers.
        If a list is given, the length must correspond to the
        number of layers. If a int is given, all layers will
        have the same size.
    n_split: int
        The number of split to create n_split linear transformations.
        In practice the input and the output are split n_split times.
        Hence we create n_split parallel linear op that will operate on
        each splited dimension. E.g. if x = [B,T,F] and n_split = 4
        then x = [B,T,4,F/4] and W = [4,F/4,out_dim/4]. This will happen
        in each layer of the VanillaNN.

    Example
    -------
    >>> inputs = torch.rand([10, 120, 60])
    >>> model = VanillaNN(input_shape=inputs.shape)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 120, 512])
    """

    def __init__(
        self,
        input_shape,
        activation: Optional[nn.Module] = torch.nn.LeakyReLU,
        dnn_blocks: Optional[int] = 2,
        dnn_neurons: Optional[int] = 512,
        n_split: Optional[int] = 1,
    ):
        super().__init__(input_shape=input_shape)

        if isinstance(dnn_neurons, list):
            if len(dnn_neurons) != dnn_blocks:
                msg = "The length of the dnn_neurons list must match dnn_blocks..."
                raise ValueError(msg)

        for block_index in range(dnn_blocks):
            if isinstance(dnn_neurons, list):
                current_nb_neurons = dnn_neurons[block_index]
            else:
                current_nb_neurons = dnn_neurons

            if n_split > 1:
                # ParrallelLinear does a costly reshape operation, hence we minimise this
                # cost by only doing this reshape for the last layer of the MLP.
                if block_index < (dnn_blocks - 1):
                    combine_out_dims = False
                else:
                    combine_out_dims = True
                self.append(
                    ParallelLinear,
                    n_neurons=current_nb_neurons,
                    bias=True,
                    n_split=n_split,
                    layer_name="linear",
                    combine_out_dims=combine_out_dims,
                )
            else:
                self.append(
                    sb.nnet.linear.Linear,
                    n_neurons=current_nb_neurons,
                    bias=True,
                    layer_name="linear",
                )
            self.append(activation(), layer_name="act")
