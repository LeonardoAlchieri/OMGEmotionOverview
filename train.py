from typing import Any
from src.utils.io import load_config
from src.utils import set_reproduction
from src.data import OMGDataset
from src.support import OmgTrainer
from logging import getLogger
import pandas as pd

logger = getLogger(__name__)


def main():
    path_to_config: str = "config_train.yaml"

    logger.info("Starting model training")
    configs: dict[str, Any] = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    set_reproduction(seed=configs["seed"], **configs["reproduction_args"])

    trainer = OmgTrainer(
        model=configs["model"],
        optimizer=configs["optimizer"],
        loss=configs["loss"],
        num_frames=configs["num_frames"],
        device=configs["device"],
        model_configs=configs["model_configs"],
        optimizer_configs=configs["optimizer_configs"],
        loss_config=configs["loss_config"],
    )

    if configs.get("backbone_weights", None) is not None:
        trainer.load_backbone_weights(
            backbone_weights_path=configs["backbone_weights"],
            strict=configs.get("backbone_weights_strict", True),
        )

    if configs.get("checkpoint", None) is not None:
        trainer.load_checkpoint(checkpoint_path=configs["checkpoint"])

    train_set = OMGDataset(
        txt_file=configs["train_list_path"],
        num_seg=configs["num_frames"],
        base_path=configs["train_data_path"],
        ground_truth_path=None,
        correct_img_size=configs["image_size"],
        transform=None,
    )

    validation_set = OMGDataset(
        txt_file=configs["validation_list_path"],
        num_seg=configs["num_frames"],
        base_path=configs["validation_data_path"],
        ground_truth_path=configs["ground_truth_path"],
        correct_img_size=configs["image_size"],
        transform=None,
    )

    history = trainer.fit(
        train_set=train_set,
        validation_set=validation_set,
        epochs=configs["epochs"],
        evaluation_frequency=configs["evaluation_frequency"],
        lr_steps=configs["lr_steps"],
        batch_size=configs["batch_size"],
        num_cpu_workers=configs["num_cpu_workers"],
        max_grad=configs["max_grad"],
        shuffle_train_set=configs["shuffle_train_set"],
    )


if __name__ == "__main__":
    main()
    print("*** TRAINING FINISHED ***")