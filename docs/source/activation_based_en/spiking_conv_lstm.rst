Spiking Convolutional LSTM
==========================

``SpikingConvLSTM`` extends :mod:`spikingjelly.activation_based.rnn` with a convolutional long short-term memory
variant that keeps activations in ``[batch, channels, height, width]`` format and iterates over temporal spike trains.
It accepts input tensors organised as ``[batch, channels, time, height, width]`` and processes them one frame at a time.

Quick start
-----------

.. code-block:: python

    import torch
    from spikingjelly.activation_based import conv_rnn, functional

    # Dummy input [batch, channels, time, height, width]
    spikes = torch.rand(2, 16, 8, 32, 32)

    model = conv_rnn.SpikingConvLSTM(
        input_channels=16,
        hidden_channels=32,
        kernel_size=3,
        num_layers=2,
        padding=1,
        dropout=0.1,
        invariant_dropout_mask=True,
    )

    output, (h_n, c_n) = model(spikes)

    # output is [batch, hidden_channels, time, height, width]
    print(output.shape)

    # ``h_n`` and ``c_n`` contain the final states for every layer (and direction, if enabled)
    print(len(h_n), h_n[0].shape)

    # Reset the dropout masks before the next simulation window when training with truncated sequences
    functional.reset_net(model)

Layer and state layout
----------------------

* ``hidden_channels`` can be a single integer shared by all layers or a list assigning a different channel count to each
  layer. The temporal dimension is preserved while the spatial resolution follows the convolutional configuration of each
  layer.
* Bidirectional mode concatenates the features from the forward and backward directions along the channel dimension.
* Initial states follow the order ``(layer 0 forward, layer 0 backward, layer 1 forward, ...)`` and should match the
  spatial size produced by the previous layer. When omitted, zero states are allocated automatically using the incoming
  frame size.

The cell-level module :class:`spikingjelly.activation_based.conv_rnn.SpikingConvLSTMCell` mirrors the behaviour of
:class:`spikingjelly.activation_based.rnn.SpikingLSTMCell` but swaps the internal linear projections with 2-D
convolutions. It is useful when building custom recurrent blocks or when integrating with modules that manage their own
state transitions.
