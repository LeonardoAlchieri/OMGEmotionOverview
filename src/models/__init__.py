import torch


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
    def __init__(self, 
                 backbone: torch.nn.Module, 
                 num_seg: int, 
                 final_layer: torch.nn.Module,
                 temporal_aggregator: torch.nn.Module | None = None, 
                 final_activation: torch.nn.Module | None = None):
        
        super(VideoEmotionRegressor, self).__init__()
        
        self.backbone = backbone
        self.num_seg = num_seg

        self.final_layer = final_layer
        self.final_activation = final_activation

        self.temporal_aggregator = temporal_aggregator

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
            raise RuntimeError(f'Expected output shape to be (batch_size, 2), but got {x.shape}. Can only predict two emotions. Check the `final_layer` variable.')
            
        return x