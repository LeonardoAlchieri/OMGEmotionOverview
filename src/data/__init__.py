
from os.path import join as join_paths

import numpy as np
import pandas as pd
import torch
from numpy.random import randint
from skimage.transform import resize
import skimage.io as io
from torch.utils.data import Dataset


class OMGDataset(Dataset):
    """
    A PyTorch dataset for loading data from the OMG-Emotion dataset.

    Parameters
    ----------
    txt_file : str
        Path to the txt file containing the metadata of the dataset.
    num_seg : int
        Number of segments per video to be extracted.
    base_path : str
        Base path to the directory containing the images of the dataset.
    correct_img_size : tuple of 3 ints, optional
        Desired size of the images in the dataset. Defaults to (112, 112, 3).
    transform : torch.nn.Module or None, optional
        Transformations to be applied to the images. Defaults to None.

    Attributes
    ----------
    base_path : str
        Base path to the directory containing the images of the dataset.
    num_seg : int
        Number of segments per video to be extracted.
    correct_img_size : tuple of 3 ints
        Desired size of the images in the dataset.
    data : pandas DataFrame
        Metadata of the dataset loaded from txt_file.
    transform : torch.nn.Module or None
        Transformations to be applied to the images.

    Methods
    -------
    __len__()
        Returns the number of samples in the dataset.
    __getitem__(idx)
        Returns the sample of the dataset at the given index.

    """

    def __init__(
        self,
        txt_file: str,
        num_seg: int,
        base_path: str,
        ground_truth_path: str | None = None,
        correct_img_size: tuple[int, int, int] = (112, 112, 3),
        transform: torch.nn.Module | None = None,
    ):
        self.base_path = base_path
        self.num_seg = num_seg
        self.correct_img_size = correct_img_size
        self.data = pd.read_csv(txt_file, sep=" ", header=0, index_col=0)
        self.data.dropna(inplace=True, how="any")
        self.transform = transform
        # not be used only in validation mode
        self.ground_truth_path = ground_truth_path

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.

        """

        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[str, str,]]:
        """
        Returns the sample of the dataset at the given index.

        Parameters
        ----------
        idx : int
            Index of the sample to be returned.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, tuple of str)
            A tuple containing the images, labels, and metadata of the sample.

        """
        vid = self.data.iloc[idx, 0]
        utter = self.data.iloc[idx, 1]
        img_list = self.data.iloc[idx, -1]
        img_list = img_list.split(",")[:-1]

        num_frames = len(img_list)
        # inspired by TSN's pytorch code
        average_duration = num_frames // self.num_seg
        if num_frames > self.num_seg:
            offsets = np.multiply(
                list(range(self.num_seg)), average_duration
            ) + randint(average_duration, size=self.num_seg)
        else:
            tick = num_frames / float(self.num_seg)
            offsets = np.array(
                [int(tick / 2.0 + tick * x) for x in range(self.num_seg)]
            )

        final_list = [img_list[i] for i in offsets]

        # stack images within a video in the depth dimension
        for i, ind in enumerate(final_list):
            image = io.imread(
                join_paths(self.base_path, "%s/%s/%s.png" % (vid, utter, ind))
            ).astype(np.float32)
            if self.correct_img_size:
                # NOTE: added here to account for possiblty different image size
                image = resize(image, self.correct_img_size, anti_aliasing=True)
            image = torch.from_numpy(((image - 127.5) / 128).transpose(2, 0, 1))

            if i == 0:
                images = image
            else:
                images = torch.cat((images, image), 0)

        label = torch.from_numpy(
            np.array([self.data.iloc[idx, 2], self.data.iloc[idx, 3]]).astype(
                np.float32
            )
        )

        if self.transform:
            image = self.transform(image)
        return (images, label, (vid, utter))
