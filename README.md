# Video-based emotion estimation using deep neural networks: a comparative study

This is the repo for the paper *Video-based emotion estimation using deep neural networks: a comparative study*. This repo contains the code used in the paper to perform training of DL models on the OMG Emotion Challenge dataset. 
Here we also give to researchers the pre-trained weights of the backbones, as used in the paper. Some are taken from other sources (see the paper for details), while some others we trained ourselves.

To run the training of the models, you should use the `train.py` script. The file `config_train.yaml` contains the necessary configurations. Specifically, it is used to specify the filepath for the dataset and the pre-trained weights of the backbone, as well as the choice for the models.

In order to run the training, the file `prepare_filelist.py` should be used. 

The code has been inspired by https://github.com/ewrfcas/OMGEmotionChallengeCode.

2023
Leonardo Alchieri, Luigi Celona, Simone Bianco

## Citation
If you use our code, please consider cite the following:
* Leonardo Alchieri, Luigi Celona, and Simone Bianco. Video-based emotion estimation using deep neural networks: a comparative study. In _ICIAP 2023 Workshops_, Springer International Publishing, 2023.
```bibtex
@inproceedings{alchieri2023video,
 author = {Alchieri, Leonardo and Celona, Luigi and Bianco},
 year = {2023},
 title = {Video-based emotion estimation using deep neural networks: a comparative study},
 booktitle = {ICIAP 2023 Workshops},
 publisher="Springer International Publishing",
}
```
