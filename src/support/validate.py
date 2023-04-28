from os.path import join as join_paths
from sys import path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

path.append("./")
from src.support.ccc import calculate_ccc

use_cuda: bool = torch.cuda.is_available()
# use_mps: bool = torch.backends.mps.is_available()
use_mps = False

# FIXME: these should not be hardcoded
# Define parameters


def validate(
    val_loader: DataLoader,
    model: torch.nn.Module,
    model_name: str,
    ground_truth_path: str,
    epoch: int = 0,
    reshape_mode: int = 1,
    device: str = 'cpu',
):
    model.eval()

    err_arou = 0.0
    err_vale = 0.0

    txt_result = open("aux_results/%s_%d.csv" % (model_name, epoch), "w")
    txt_result.write("video,utterance,arousal,valence\n")
    for inputs, targets, (vid, utter) in tqdm(val_loader, "Validation batch"):
        inputs: Tensor
        targets: Tensor
        
        inputs, targets = inputs.to(device), targets.to(device)

        # NOTE: added for way resnet wants shape
        if reshape_mode == 1:
            inputs = inputs.reshape(
                inputs.shape[0], -1, 3, inputs.shape[-2], inputs.shape[-1]
            )
        elif reshape_mode == 2:
            inputs = inputs.view((-1, 3) + inputs.size()[-2:])
        else:
            raise ValueError("reshape_mode must be 1 or 2. Got %d" % reshape_mode)
        outputs: Tensor
        outputs = model(inputs)

        outputs = outputs.data.cpu().numpy()
        targets = targets.data.cpu().numpy()

        err_arou += np.sum((outputs[:, 0] - targets[:, 0]) ** 2)
        err_vale += np.sum((outputs[:, 1] - targets[:, 1]) ** 2)

        for i in range(len(vid)):
            out = outputs
            txt_result.write(
                "%s,%s.mp4,%f,%f\n" % (vid[i], utter[i], out[i][0], out[i][1])
            )

    txt_result.close()

    arouCCC, valeCCC = calculate_ccc(
        ground_truth_path,
        "aux_results/%s_%d.csv" % (model_name, epoch),
    )
    return (arouCCC, valeCCC)
