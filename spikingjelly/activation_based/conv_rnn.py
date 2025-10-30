"""Convolutional recurrent building blocks for activation-based spiking networks."""

import math
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from . import layer, surrogate


def _ensure_sequence(value: Union[int, Sequence], num_layers: int, name: str) -> List:
    """Broadcast a value to match ``num_layers`` when necessary."""

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        value_list = list(value)
        if num_layers == 1:
            return [value]
        if len(value_list) == num_layers:
            return value_list
        return [value for _ in range(num_layers)]
    return [value for _ in range(num_layers)]


class SpikingConvLSTMCell(nn.Module):
    r"""A convolutional variant of :class:`~spikingjelly.activation_based.rnn.SpikingLSTMCell`.

    The cell keeps inputs, hidden states and cell states in the spatial layout ``[batch, channels, height, width]`` and
    replaces the internal affine projections with two-dimensional convolutions. The gating logic is identical to the
    dense spiking LSTM:

    .. math::

        \begin{align}
        i &= \Theta\left(W_{xi} * x + b_{xi} + W_{hi} * h + b_{hi}\right) \\\\
        f &= \Theta\left(W_{xf} * x + b_{xf} + W_{hf} * h + b_{hf}\right) \\\\
        g &= \Theta\left(W_{xg} * x + b_{xg} + W_{hg} * h + b_{hg}\right) \\\\
        o &= \Theta\left(W_{xo} * x + b_{xo} + W_{ho} * h + b_{ho}\right) \\\\
        c' &= f \odot c + i \odot g \\\\
        h' &= o \odot c'
        \end{align}

    where ``*`` denotes a convolution and ``\Theta`` is the spiking activation specified by the surrogate function.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        hidden_kernel_size: Optional[Union[int, Tuple[int, int]]] = None,
        hidden_padding: Optional[Union[int, Tuple[int, int]]] = None,
        hidden_dilation: Optional[Union[int, Tuple[int, int]]] = None,
        surrogate_function1: surrogate.SurrogateFunctionBase = surrogate.Erf(),
        surrogate_function2: Optional[surrogate.SurrogateFunctionBase] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.surrogate_function1 = surrogate_function1
        self.surrogate_function2 = surrogate_function2
        if self.surrogate_function2 is not None:
            if self.surrogate_function1.spiking != self.surrogate_function2.spiking:
                raise ValueError("surrogate_function1 and surrogate_function2 must share the same spiking flag")

        k_in = _pair(kernel_size)
        k_hidden = _pair(hidden_kernel_size if hidden_kernel_size is not None else kernel_size)
        p_in = _pair(padding)
        p_hidden = _pair(hidden_padding if hidden_padding is not None else padding)
        d_in = _pair(dilation)
        d_hidden = _pair(hidden_dilation if hidden_dilation is not None else dilation)
        s_in = _pair(stride)

        self.conv_x = layer.Conv2d(
            in_channels,
            4 * hidden_channels,
            k_in,
            stride=s_in,
            padding=p_in,
            dilation=d_in,
            bias=bias,
        )
        self.conv_h = layer.Conv2d(
            hidden_channels,
            4 * hidden_channels,
            k_hidden,
            stride=1,
            padding=p_hidden,
            dilation=d_hidden,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        sqrt_k = math.sqrt(1.0 / self.hidden_channels)
        for param in self.parameters():
            nn.init.uniform_(param, -sqrt_k, sqrt_k)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"expected x with shape [batch, channels, height, width], but got {x.shape}")

        if state is None:
            h = torch.zeros(
                (x.shape[0], self.hidden_channels, x.shape[2], x.shape[3]),
                dtype=x.dtype,
                device=x.device,
            )
            c = torch.zeros_like(h)
        else:
            h, c = state

        gates = self.conv_x(x) + self.conv_h(h)

        if self.surrogate_function2 is None:
            i, f, g, o = torch.chunk(self.surrogate_function1(gates), 4, dim=1)
        else:
            i, f, g, o = torch.chunk(gates, 4, dim=1)
            i = self.surrogate_function1(i)
            f = self.surrogate_function1(f)
            g = self.surrogate_function2(g)
            o = self.surrogate_function1(o)

        c = c * f + i * g
        with torch.no_grad():
            torch.clamp_max_(c, 1.0)
        h = c * o
        return h, c


class SpikingConvLSTM(nn.Module):
    """Stacked convolutional spiking LSTM layers that operate on ``[batch, channels, time, height, width]`` inputs."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: Union[int, Sequence[int]],
        kernel_size: Union[int, Sequence[Union[int, Tuple[int, int]]]],
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        invariant_dropout_mask: bool = False,
        bidirectional: bool = False,
        stride: Union[int, Sequence[Union[int, Tuple[int, int]]]] = 1,
        padding: Union[int, Sequence[Union[int, Tuple[int, int]]]] = 0,
        dilation: Union[int, Sequence[Union[int, Tuple[int, int]]]] = 1,
        hidden_kernel_size: Optional[Union[int, Sequence[Union[int, Tuple[int, int]]]]] = None,
        hidden_padding: Optional[Union[int, Sequence[Union[int, Tuple[int, int]]]]] = None,
        hidden_dilation: Optional[Union[int, Sequence[Union[int, Tuple[int, int]]]]] = None,
        surrogate_function1: surrogate.SurrogateFunctionBase = surrogate.Erf(),
        surrogate_function2: Optional[surrogate.SurrogateFunctionBase] = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers has to be >= 1")
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.dropout = float(dropout)
        self.invariant_dropout_mask = invariant_dropout_mask

        hidden_channels_list = _ensure_sequence(hidden_channels, num_layers, "hidden_channels")
        kernel_size_list = _ensure_sequence(kernel_size, num_layers, "kernel_size")
        stride_list = _ensure_sequence(stride, num_layers, "stride")
        padding_list = _ensure_sequence(padding, num_layers, "padding")
        dilation_list = _ensure_sequence(dilation, num_layers, "dilation")
        hidden_kernel_list = _ensure_sequence(hidden_kernel_size, num_layers, "hidden_kernel_size") if hidden_kernel_size is not None else [None] * num_layers
        hidden_padding_list = _ensure_sequence(hidden_padding, num_layers, "hidden_padding") if hidden_padding is not None else [None] * num_layers
        hidden_dilation_list = _ensure_sequence(hidden_dilation, num_layers, "hidden_dilation") if hidden_dilation is not None else [None] * num_layers

        cells: List[nn.ModuleList] = []
        in_channels = input_channels
        for layer_idx in range(num_layers):
            layer_cells = nn.ModuleList()
            for direction in range(self.num_directions):
                cell = SpikingConvLSTMCell(
                    in_channels,
                    hidden_channels_list[layer_idx],
                    kernel_size_list[layer_idx],
                    stride=stride_list[layer_idx],
                    padding=padding_list[layer_idx],
                    dilation=dilation_list[layer_idx],
                    bias=bias,
                    hidden_kernel_size=hidden_kernel_list[layer_idx],
                    hidden_padding=hidden_padding_list[layer_idx],
                    hidden_dilation=hidden_dilation_list[layer_idx],
                    surrogate_function1=surrogate_function1,
                    surrogate_function2=surrogate_function2,
                )
                layer_cells.append(cell)
            cells.append(layer_cells)
            in_channels = hidden_channels_list[layer_idx] * self.num_directions

        self.layers = nn.ModuleList(cells)
        self.hidden_channels_list = hidden_channels_list

        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        self.register_buffer("_dropout_masks", None, persistent=False)

    def reset(self) -> None:
        """Clear cached dropout masks so that :func:`spikingjelly.activation_based.functional.reset_net` works."""
        self._dropout_masks = None

    def _apply_dropout(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if self.dropout <= 0 or layer_idx == self.num_layers - 1:
            return x
        if not self.training:
            return x
        if self.invariant_dropout_mask:
            if self._dropout_masks is None:
                self._dropout_masks = [None for _ in range(self.num_layers - 1)]
            mask = self._dropout_masks[layer_idx]
            if mask is None or mask.shape != x.shape[1:]:
                mask = F.dropout(torch.ones_like(x[0]), self.dropout, training=True)
                self._dropout_masks[layer_idx] = mask
            return x * mask.unsqueeze(0)
        return F.dropout(x, p=self.dropout, training=True)

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        if x.dim() != 5:
            raise ValueError(
                f"SpikingConvLSTM expects inputs with shape [batch, channels, time, height, width], but received {x.shape}"
            )
        seq = x.permute(2, 0, 1, 3, 4)  # [T, B, C, H, W]

        if states is None:
            h_states: List[Optional[torch.Tensor]] = [None] * (self.num_layers * self.num_directions)
            c_states: List[Optional[torch.Tensor]] = [None] * (self.num_layers * self.num_directions)
        else:
            if not isinstance(states, (tuple, list)) or len(states) != 2:
                raise ValueError("states should be a tuple (h_0, c_0)")
            h_states, c_states = list(states[0]), list(states[1])
            if len(h_states) != self.num_layers * self.num_directions or len(c_states) != self.num_layers * self.num_directions:
                raise ValueError("states must provide tensors for every layer and direction")

        layer_input = seq
        final_h: List[torch.Tensor] = []
        final_c: List[torch.Tensor] = []
        for layer_idx in range(self.num_layers):
            outputs_per_direction = []
            time_steps = layer_input.shape[0]
            for direction in range(self.num_directions):
                cell = self.layers[layer_idx][direction]
                current_sequence = layer_input if direction == 0 else torch.flip(layer_input, dims=[0])
                state_index = layer_idx * self.num_directions + direction
                h_c_tuple: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                if h_states[state_index] is not None and c_states[state_index] is not None:
                    h_c_tuple = (h_states[state_index], c_states[state_index])

                outputs = []
                for t in range(time_steps):
                    h_c_tuple = cell(current_sequence[t], h_c_tuple)
                    outputs.append(h_c_tuple[0])
                out_seq = torch.stack(outputs, dim=0)
                if direction == 1:
                    out_seq = torch.flip(out_seq, dims=[0])
                outputs_per_direction.append(out_seq)
                final_h.append(h_c_tuple[0])
                final_c.append(h_c_tuple[1])
            layer_output = torch.cat(outputs_per_direction, dim=2)
            layer_output = self._apply_dropout(layer_output, layer_idx)
            layer_input = layer_output

        output = layer_input.permute(1, 2, 0, 3, 4)
        return output, (final_h, final_c)
