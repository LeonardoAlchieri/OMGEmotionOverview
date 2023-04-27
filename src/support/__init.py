from typing import Any
import torch
from torchvision.models import convnext_small

from src.models import VideoEmotionRegressor, FinalActivation, TemporalAggregator
from src.models.swine import dualHead, swin_3d_tiny
from src.models.net_sphere import sphere20a
from src.models.our_backbones import facebval
from src.models.former_dfer import FormerDfer


class OmgTrainer:
    def __init__(
        self,
        model: str | torch.nn.Module,
        optimizer: str | torch.nn.Module,
        loss: str | torch.nn.Module,
        num_frames: int,
        model_params: dict[str, Any] = {},
    ):
        self.model = (
            self._get_model(
                model_name=model, model_params=model_params, num_frames=num_frames
            )
            if isinstance(model, str)
            else model
        )
        self.optimizer = optimizer if isinstance(optimizer, str) else optimizer
        self.loss = loss if isinstance(loss, str) else loss

    @staticmethod
    def _get_model(
        model_name: str, num_frames: int, model_params: dict[str, Any]
    ) -> torch.nn.Module:
        temporal_aggregator = model_params.get("temporal_aggregator", None)

        accepted_activations: list[str] = [
            "resnet-50",
            "sphereface20",
            "former-dfer",
            "swine3dtiny",
            "convnext-small",
        ]
        match model_name:
            case "resnet-50":
                final_layer = torch.nn.Linear(
                    model_params.get("temporal_aggregation_output_size", 512), 2
                )
                backbone = facebval()
                if temporal_aggregator is not None:
                    temporal_aggregator = TemporalAggregator(
                        aggregator=temporal_aggregator,
                        backbone_output_size=2048,
                        num_seg=num_frames,
                        backbone_output_size=model_params.get(
                            "temporal_aggregation_output_size", 512
                        ),
                    )
            case "sphereface20":
                final_layer = torch.nn.Linear(
                    model_params.get("temporal_aggregation_output_size", 512), 2
                )
                backbone = sphere20a(classnum=10574, feature=True)
                if temporal_aggregator is not None:
                    temporal_aggregator = TemporalAggregator(
                        aggregator=temporal_aggregator,
                        backbone_output_size=512,
                        num_seg=num_frames,
                        backbone_output_size=model_params.get(
                            "temporal_aggregation_output_size", 512
                        ),
                    )
            case "former-dfer":
                final_layer = torch.nn.Linear(
                    model_params.get("temporal_aggregation_output_size", 512), 2
                )
                if temporal_aggregator == "transformer":
                    backbone = FormerDfer(use_temporal_part=True)
                    temporal_aggregator = None
                else:
                    backbone = FormerDfer(use_temporal_part=False)
            case "swine3dtiny":
                final_layer = dualHead(768, 64)
                backbone = swin_3d_tiny()
            case "convnext-small":
                final_layer = torch.nn.Linear(
                    model_params.get("temporal_aggregation_output_size", 512), 2
                )
                backbone = convnext_small(weights=True)
                if temporal_aggregator is not None:
                    temporal_aggregator = TemporalAggregator(
                        aggregator=temporal_aggregator,
                        backbone_output_size=1000,
                        num_seg=num_frames,
                        backbone_output_size=model_params.get(
                            "temporal_aggregation_output_size", 512
                        ),
                    )
            case _:
                raise ValueError(
                    f"Invalid model name. Accepted names are {accepted_activations}"
                )

        final_activation = model_params.get("final_activation", None)
        if final_activation is not None:
            final_activation = FinalActivation(activation=final_activation)

        return VideoEmotionRegressor(
            backbone=backbone,
            num_seg=num_frames,
            final_layer=final_layer,
            temporal_aggregator=temporal_aggregator,
            final_activation=final_activation,
        )
