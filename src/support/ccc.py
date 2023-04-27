from __future__ import print_function

import argparse
import os
import sys
from typing import Tuple

import torch
import numpy
import pandas
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, mean_squared_error


def mse(y_true: torch.Tensor | numpy.ndarray, y_pred: torch.Tensor | numpy.ndarray):
    """
    Calculates the mean squared error between two arrays.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    mse : float
        Mean squared error between y_true and y_pred.
    """
    return mean_squared_error(y_true, y_pred)


def f1(y_true: torch.Tensor | numpy.ndarray, y_pred: torch.Tensor | numpy.ndarray):
    """
    Computes the F1 score between two arrays.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    f1_score : float
        F1 score between y_true and y_pred.
    """
    label = [0, 1, 2, 3, 4, 5, 6]
    return f1_score(y_true, y_pred, labels=label, average="micro")


def ccc(y_true: torch.Tensor | numpy.ndarray, y_pred: torch.Tensor | numpy.ndarray):
    """
    Calculates the concordance correlation coefficient (CCC) and Pearson correlation coefficient between two arrays.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    ccc : float
        Concordance correlation coefficient between y_true and y_pred.
    rho : float
        Pearson correlation coefficient between y_true and y_pred.
    """
    true_mean = numpy.mean(y_true)
    true_variance = numpy.var(y_true)
    pred_mean = numpy.mean(y_pred)
    pred_variance = numpy.var(y_pred)

    rho, _ = pearsonr(y_pred, y_true)

    std_predictions = numpy.std(y_pred)

    std_gt = numpy.std(y_true)

    ccc = (
        2
        * rho
        * std_gt
        * std_predictions
        / (std_predictions**2 + std_gt**2 + (pred_mean - true_mean) ** 2)
    )

    return ccc, rho


def calculateCCC(validationFile: str, modelOutputFile: str):
    """
    Calculates the CCC, Pearson correlation coefficient, and mean squared error between the ground truth and estimated
    target values.

    Parameters
    ----------
    validationFile : str
        Path to the CSV file containing the ground truth target values.
    modelOutputFile : str
        Path to the CSV file containing the estimated target values.

    Returns
    -------
    arousalCCC : float
        Concordance correlation coefficient between the arousal values in the ground truth and estimated target arrays.
    valenceCCC : float
        Concordance correlation coefficient between the valence values in the ground truth and estimated target arrays.
    """
    dataY = pandas.read_csv(validationFile, header=0, sep=",")

    dataYPred = pandas.read_csv(modelOutputFile, header=0, sep=",")

    dataYArousal = dataY["arousal"]
    dataYValence = dataY["valence"]
    dataYPredArousal = dataYPred["arousal"]
    dataYPredValence = dataYPred["valence"]

    arousalCCC, acor = ccc(dataYArousal, dataYPredArousal)
    arousalmse = mse(dataYArousal, dataYPredArousal)
    valenceCCC, vcor = ccc(dataYValence, dataYPred)
    valencemse = mse(dataYValence, dataYPredValence)

    print("Arousal CCC: ", arousalCCC)
    print("Arousal Pearson Cor: ", acor)
    print("Arousal MSE: ", arousalmse)
    print("Valence CCC: ", valenceCCC)
    print("Valence cor: ", vcor)
    print("Valence MSE: ", valencemse)
    return arousalCCC, valenceCCC
