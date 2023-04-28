import torch


class TemporalAggregator(torch.nn.Module):
    
    __name__ = "general_aggregator"
    def __init__(
        self, aggregator_name: str, num_seg: int, backbone_output_size: int, **kwargs
    ):
        super(TemporalAggregator, self).__init__()

        self.avgPool = torch.nn.AvgPool2d((num_seg, 1), stride=1)
        self.num_seg = num_seg
        match aggregator_name:
            case "bi-lstm":
                self.aggregator = torch.nn.LSTM(
                    backbone_output_size,
                    kwargs.get("output_size", 512),
                    1,
                    batch_first=kwargs.get("batch_first", True),
                    dropout=kwargs.get("dropout", 0.2),
                    bidirectional=True,
                )  # Input dim, hidden dim, num_layer
            case "bi-gru":
                self.aggregator = torch.nn.GRU(
                    backbone_output_size,
                    kwargs.get("output_size", 512),
                    1,
                    batch_first=kwargs.get("batch_first", True),
                    dropout=kwargs.get("dropout", 0.2),
                    bidirectional=kwargs.get("bidirectional", True),
                )  # Input dim, hidden dim, num_layer
            case "lstm":
                self.aggregator = torch.nn.LSTM(
                    backbone_output_size,
                    kwargs.get("output_size", 512),
                    1,
                    batch_first=kwargs.get("batch_first", True),
                    dropout=kwargs.get("dropout", 0.2),
                    bidirectional=False,
                )  # Input dim, hidden dim, num_layer
            case "gru":
                self.aggregator = torch.nn.GRU(
                    backbone_output_size,
                    kwargs.get("output_size", 512),
                    1,
                    batch_first=kwargs.get("batch_first", True),
                    dropout=kwargs.get("dropout", 0.2),
                    bidirectional=False,
                )  # Input dim, hidden dim, num_layer
            case _:
                raise ValueError(f"Invalid temporal aggregator: {aggregator_name}")

        self.__name__ = aggregator_name

        if kwargs.get("init_weights", True):
            for name, param in self.aggregator.named_parameters():
                if "bias" in name:
                    torch.nn.init.constant(param, 0.0)
                elif "weight" in name:
                    torch.nn.init.orthogonal(param)

    def sequentialLSTM(self, input: torch.Tensor, hidden=None) -> torch.Tensor:
        input_lstm = input.view([-1, self.num_seg, input.shape[1]])
        batch_size = input_lstm.shape[0]
        feature_size = input_lstm.shape[2]

        self.aggregator.flatten_parameters()

        output_lstm, hidden = self.aggregator(input_lstm)

        if self.aggregator.bidirectional:
            output_lstm = (
                output_lstm.contiguous()
                .view(batch_size, output_lstm.size(1), 2, -1)
                .sum(2)
                .view(batch_size, output_lstm.size(1), -1)
            )

        # avarage the output of LSTM
        output_lstm = output_lstm.view(batch_size, 1, self.num_seg, -1)
        out = self.avgPool(output_lstm)
        out = out.view(batch_size, -1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequentialLSTM(x)


class FinalActivation(torch.nn.Module):
    
    __name__ = "default_activation"
    def __init__(self, activation: str):
        super(FinalActivation, self).__init__()

        if activation == "tanh-sigmoid":
            self.valence_activation = torch.nn.Tanh()
            self.arousal_activation = torch.nn.Sigmoid()
        elif activation == "double-tanh":
            self.valence_activation = torch.nn.Tanh()
            self.arousal_activation = torch.nn.Tanh()
        elif activation == "hard-tanh":
            self.valence_activation = torch.nn.Hardtanh(min_value=-1, max_value=1)
            self.arousal_activation = torch.nn.Hardtanh(min_value=0, max_value=1)
        else:
            raise ValueError(f"Invalid activation function: {activation}")

        self.__name__ = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        valence = self.valence_activation(x[:, 0])
        arousal = self.arousal_activation(x[:, 1])

        return torch.stack((valence, arousal), dim=1)


class VideoEmotionRegressor(torch.nn.Module):
    """
    A neural network model for predicting the emotion of a video, based on a given backbone architecture.

    Parameters
    ----------
    backbone : torch.nn.Module
        The backbone architecture of the model. This should be a pre-trained neural network that can process
        video data, such as a convolutional neural network (CNN).
    num_seg : int
        Number of frames the video is mode of. The frames can also be sampled from
        the video, and not be consecutive, to reduce overhead.
    final_layer : torch.nn.Module
        The final layer of the model. This should be a neural network layer that can map the output of the backbone
        architecture to the predicted emotions. Note that this layer should output a tensor of shape (batch_size, 2),
        where each element in the second dimension represents the predicted probability of a specific emotion (e.g.
        happiness and sadness).
    temporal_aggregator : torch.nn.Module, optional
        An optional neural network layer that can aggregate the output of the backbone architecture over time. This
        can be useful for processing long videos or for capturing temporal dynamics in the data.
    final_activation : torch.nn.Module, optional
        An optional activation function to apply to the output of the final layer.
        For example, tanh or sigmoid.

    Methods
    -------
    forward(x)
        Compute the forward pass of the model.

    Raises
    ------
    RuntimeError
        If the shape of the final layer output is not (batch_size, 2), indicating an error in the model architecture.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        num_seg: int,
        final_layer: torch.nn.Module,
        temporal_aggregator: torch.nn.Module | None = None,
        final_activation: torch.nn.Module | None = None,
    ):
        super(VideoEmotionRegressor, self).__init__()

        self.backbone = backbone
        self.num_seg = num_seg

        self.final_layer = final_layer
        self.final_activation = final_activation

        self.temporal_aggregator = temporal_aggregator
        self.__name__ = f"{backbone.__name__}_{temporal_aggregator.__name__}_{final_activation.__name__}_{num_seg}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input video data, with shape (batch_size, num_channels, num_frames, height, width).

        Returns
        -------
        torch.Tensor
            The predicted emotion probabilities, with shape (batch_size, 2).
        """
        x = self.backbone(x)
        if self.temporal_aggregator is not None:
            x = self.temporal_aggregator(x)

        x = self.final_layer(x)

        if self.final_activation is not None:
            x = self.final_activation(x)

        if x.shape[1] != 2:
            raise RuntimeError(
                f"Expected output shape to be (batch_size, 2), but got {x.shape}. Can only predict two emotions. Check the `final_layer` variable."
            )

        return x
