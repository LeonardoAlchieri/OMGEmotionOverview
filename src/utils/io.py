from glob import glob
from json import dumps as json_dumps
from logging import getLogger
from os import getlogin, makedirs, mkdir
from os.path import isdir, isfile
from os.path import join as join_paths
from shutil import copyfile
from typing import Any, Dict, List

import torch
from execution_time_wrapper import get_execution_time_print
from numpy import arange
from pandas import DataFrame
from yaml import safe_load as load_yaml

logger = getLogger("utils/io")


@get_execution_time_print
def load_config(path: str) -> dict[str, Any]:
    """Simple method to load yaml configuration for a given script.

    Args:
        path (str): path to the yaml file

    Returns:
        Dict[str, Any]: the method returns a dictionary with the loaded configurations
    """
    with open(path, "r") as file:
        config_params = load_yaml(file)
    return config_params


def save_model(model: torch.nn.Module, filename: str) -> None:
    state = model.state_dict()
    torch.save(state, filename)


def check_create_folder(path: str) -> None:
    """This method can be used to check if a folder exists and create it if it doesn't.

    Args:
        path (str): path to the folder to be checked and created
    """
    if isdir(path):
        logger.debug(f"Path exists. Not creating it.")
    elif isfile(path):
        raise RuntimeError(f"The output path {path} exists but is not a folder.")
    else:
        logger.warning(f"Creating output folder {path}.")
        makedirs(path)


def create_output_folder(
    path_to_config: str, task: str, model: str, run: str = "train"
) -> str:
    """This method can be used to create the output folder for the current task.

    Parameters
    ----------
    path_to_config : str
        path to the model configuration file, which will be copied with
        the rest of the information
    task : str
        task name
    model : str
        model name
    run : str
        identifies if the current run is a train or test run

    Returns
    -------
    str
        the model returns the relative session output path

    Raises
    ------
    RuntimeError
        if the path for the model exists as a file, an error in thrown
    RuntimeError
        if the current identified session exists, an error is thrown
    """
    namespace: str = getlogin()
    base_output_path: str = f"./results.nosync/{run}/"
    given_output_path: str = join_paths(base_output_path, task, model)
    logger.info(f"General output path: {given_output_path}")
    check_create_folder(path=given_output_path)

    current_sessions_ids: List[int] = [
        int("".join([char for char in session.split("/")[-1] if char.isdigit()]))
        for session in glob(join_paths(given_output_path, f"{namespace}session*"))
    ]
    if len(current_sessions_ids) == 0:
        current_session: int = 0
    else:
        last_session: int = max(current_sessions_ids)
        logger.debug(f"Last session: {last_session}")
        current_session: int = last_session + 1
    logger.debug(f"Current session: {current_session}")
    current_session_path: str = join_paths(
        given_output_path, f"{namespace}session{current_session}"
    )
    logger.info(f"Saving current session to {current_session_path}")
    if isdir(current_session_path):
        raise RuntimeError(f"The output path {current_session_path} already exists.")
    elif isfile(current_session_path):
        raise RuntimeError(
            f"The output path {current_session_path} exists but is not a folder."
        )
    else:
        mkdir(current_session_path)

    copyfile(path_to_config, join_paths(current_session_path, "config.yaml"))
    return current_session_path


def save_history(
    history: list[dict[str, float]],
    output_name: str,
) -> None:
    """This method can be used to save the training history of a model.

    Args:
        history (list[dict[str, float]]): list of dictionaries with the training history
        output_name (str): name of the file to be saved
    """
    # save to json the list of dict
    with open(output_name, "w") as file:
        file.write(json_dumps(history))
