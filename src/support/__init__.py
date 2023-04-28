from typing import Any, Iterable
import torch
from torchvision.models import convnext_small
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from src.models import VideoEmotionRegressor, FinalActivation, TemporalAggregator
from src.models.swine import dualHead, swin_3d_tiny
from src.models.net_sphere import sphere20a
from src.models.our_backbones import facebval
from src.models.former_dfer import FormerDfer
from src.models.dan import DAN
from src.utils.loss import VALoss
from src.support.train import train
from src.support.validate import validate
from src.data import OMGDataset
from src.utils.io import save_model
from src.utils import load_backbone_weight


ACCEPTED_MODEL_NAMES: list[str] = [
    "resnet-50",
    "sphereface20",
    "former-dfer",
    "swine3dtiny",
    "convnext-small",
    "dan",
]


class OmgTrainer:
    def __init__(
        self,
        model: str | torch.nn.Module,
        optimizer: str | torch.nn.Module,
        loss: str | torch.nn.Module,
        num_frames: int,
        device: str = "cpu",
        model_configs: dict[str, Any] = {},
        optimizer_configs: dict[str, Any] = {},
        loss_config: dict[str, Any] = {},
    ):
        if isinstance(model, str):
            self.model_name: str = model
        else:
            self.model_name: str = model.__name__
        self.model: VideoEmotionRegressor = (
            self._get_model(
                model_name=model, model_configs=model_configs, num_frames=num_frames
            )
            if isinstance(model, str)
            else model
        )
        self.optimizer: torch.optim.Optimizer = (
            self._get_optimizer(
                optimizer_name=optimizer,
                model_params=self.model.parameters(),
                optimizer_configs=optimizer_configs,
            )
            if isinstance(optimizer, str)
            else optimizer
        )
        self.loss: torch.nn.Module = (
            self._get_loss(loss_name=loss, loss_config=loss_config)
            if isinstance(loss, str)
            else loss
        )

        self.device = device
        self.pre_trained: bool = False

        self.model = self.model.to(self.device)

    @staticmethod
    def _get_model(
        model_name: str, num_frames: int, model_configs: dict[str, Any]
    ) -> torch.nn.Module:
        temporal_aggregator = model_configs.get("temporal_aggregator", None)

        match model_name:
            case "resnet-50":
                final_layer = torch.nn.Linear(
                    model_configs.get("temporal_aggregation_output_size", 512), 2
                )
                backbone = facebval()
                if temporal_aggregator is not None:
                    temporal_aggregator = TemporalAggregator(
                        aggregator_name=temporal_aggregator,
                        backbone_output_size=2048,
                        num_seg=num_frames,
                        output_size=model_configs.get(
                            "temporal_aggregation_output_size", 512
                        ),
                    )
            case "sphereface20":
                final_layer = torch.nn.Linear(
                    model_configs.get("temporal_aggregation_output_size", 512), 2
                )
                backbone = sphere20a(classnum=10574, feature=True)
                if temporal_aggregator is not None:
                    temporal_aggregator = TemporalAggregator(
                        aggregator_name=temporal_aggregator,
                        backbone_output_size=512,
                        num_seg=num_frames,
                        output_size=model_configs.get(
                            "temporal_aggregation_output_size", 512
                        ),
                    )
            case "former-dfer":
                final_layer = torch.nn.Linear(
                    model_configs.get("temporal_aggregation_output_size", 512), 2
                )
                if temporal_aggregator == "transformer":
                    backbone = FormerDfer(use_temporal_part=True)
                    temporal_aggregator = None
                else:
                    backbone = FormerDfer(use_temporal_part=False)
            case "dan":
                final_layer = torch.nn.Linear(
                    model_configs.get("temporal_aggregation_output_size", 512), 2
                )
                backbone = DAN(num_class=8)
                if temporal_aggregator is not None:
                    temporal_aggregator = TemporalAggregator(
                        aggregator_name=temporal_aggregator,
                        backbone_output_size=512,
                        num_seg=num_frames,
                        output_size=model_configs.get(
                            "temporal_aggregation_output_size", 512
                        ),
                    )
            case "swine3dtiny":
                final_layer = dualHead(768, 64)
                backbone = swin_3d_tiny()
            case "convnext-small":
                final_layer = torch.nn.Linear(
                    model_configs.get("temporal_aggregation_output_size", 512), 2
                )
                backbone = convnext_small(weights=True)
                if temporal_aggregator is not None:
                    temporal_aggregator = TemporalAggregator(
                        aggregator_name=temporal_aggregator,
                        backbone_output_size=1000,
                        num_seg=num_frames,
                        output_size=model_configs.get(
                            "temporal_aggregation_output_size", 512
                        ),
                    )
            case _:
                raise ValueError(
                    f"Invalid model name. Accepted names are {ACCEPTED_MODEL_NAMES}"
                )

        final_activation = model_configs.get("final_activation", None)
        if final_activation is not None:
            final_activation = FinalActivation(activation=final_activation)

        return VideoEmotionRegressor(
            backbone=backbone,
            num_seg=num_frames,
            final_layer=final_layer,
            temporal_aggregator=temporal_aggregator,
            final_activation=final_activation,
        )

    @staticmethod
    def _get_optimizer(
        optimizer_name: str, model_params: Iterable, optimizer_configs: dict[str, Any]
    ) -> torch.optim.Optimizer:
        accepted_optimizers: list[str] = ["SGD"]
        match optimizer_name:
            case "SGD":
                return torch.optim.SGD(
                    params=model_params,
                    lr=optimizer_configs.get("learning_rate", 0.01),
                    momentum=optimizer_configs.get("momentum", 0.9),
                    weight_decay=optimizer_configs.get("weight_decay", 5e-4),
                )
            case _:
                raise ValueError(
                    f"Invalid optimizer name. Accepted names are {accepted_optimizers}"
                )

    @staticmethod
    def _get_loss(loss_name: str, loss_config: dict[str, Any] = {}) -> torch.nn.Module:
        accepted_losses: list[str] = ["CCC", "MSE"]

        match loss_name:
            case "CCC":
                return VALoss(
                    loss_type="CCC",
                    digitize_num=1,
                    val_range=[-1, 1],
                    aro_range=[0, 1],
                    lambda_ccc=loss_config.get("lambda_ccc", 2),
                    lambda_v=loss_config.get("lambda_v", 1),
                    lambda_a=loss_config.get("lambda_a", 1),
                )
            case "MSE":
                return torch.nn.MSELoss()
            case _:
                raise ValueError(
                    f"Invalid loss name. Accepted names are {accepted_losses}"
                )

    def _update_learning_rate(self, new_lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def _get_reshape_mode(self) -> int:
        match self.model_name:
            case "resnet-50" | "former-dfer" | "dan" | "swine3dtiny":
                return 1
            case "sphereface20" | "convnext-small":
                return 2
            case _:
                raise ValueError(
                    f"Invalid model name. Accepted names are {ACCEPTED_MODEL_NAMES}. Received: {self.model_name}"
                )

    def _get_config_name(self) -> str:
        if self.pre_trained:
            return f"{self.model.__name__}_{self.loss.__class__.__name__}_{self.optimizer.__class__.__name__}_pretrained"
        else:
            return f"{self.model.__name__}_{self.loss.__class__.__name__}_{self.optimizer.__class__.__name__}"

    def fit(
        self,
        train_set: OMGDataset,
        validation_set: OMGDataset,
        epochs: int = 1,
        evaluation_frequency: int = 1,
        lr_steps: list[int] = [8, 16, 24],
        batch_size: int = 1,
        num_cpu_workers: int = 1,
        max_grad: float = 20.0,
        shuffle_train_set: bool = True,
    ) -> list[dict[str, float]]:
        if not isinstance(train_set, Dataset) or not isinstance(
            validation_set, Dataset
        ):
            raise ValueError(
                f"train_set and validation_set must be of type Dataset. Received {type(train_set)} and {type(validation_set)} respectively."
            )
        else:
            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=shuffle_train_set,
                num_workers=num_cpu_workers,
            )
            val_loader = DataLoader(
                validation_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_cpu_workers,
            )

        for epoch in tqdm(range(epochs), desc="Epoch"):
            if epoch in lr_steps:
                lr *= 0.1
                self._update_learning_rate(new_lr=lr)

            train(
                train_loader=train_loader,
                model=self.model,
                criterion=self.loss,
                optimizer=self.optimizer,
                epoch=epoch,
                reshape_mode=self._get_reshape_mode(),
                device=self.device,
                max_grad=max_grad,
                print_freq=18,
            )

            history = []
            # evaluate on validation set
            if (epoch + 1) % evaluation_frequency == 0 or epoch == epochs - 1:
                arou_ccc, vale_ccc = validate(
                    val_loader=val_loader,
                    model=self.model,
                    model_name=self.model_name,
                    epoch=epoch,
                    device=self.device,
                    ground_truth_path=validation_set.ground_truth_path,
                    reshape_mode=self._get_reshape_mode(),
                )

                history.append(
                    {"epoch": epoch, "arousal CCC": arou_ccc, "valence CCC": vale_ccc}
                )

                if (arou_ccc + vale_ccc) > (best_arou_ccc + best_vale_ccc):
                    best_arou_ccc = arou_ccc
                    best_vale_ccc = vale_ccc
                    save_model(
                        self.model,
                        (
                            "./pth/%s_%i_%.4f_%.4f.pth"
                            % (self._get_config_name(), epoch, arou_ccc, vale_ccc)
                        ),
                    )

        return history

    def load_backbone_weights(
        self, backbone_weights_path: str, strict: bool = True
    ) -> None:
        self.model.backbone.load_state_dict(
            load_backbone_weight(
                weights_path=backbone_weights_path, loading_device=self.device
            ),
            strict=strict,
        )  # while should not be used, I tested and only the final prediction layers are indeed missing
        self.pre_trained = True

    def load_checkpoint(self, checkpoint_path: str) -> None:
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device), strict=True
        )
