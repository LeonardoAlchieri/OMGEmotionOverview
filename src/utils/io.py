from typing import Any
from yaml import load as load_yaml

from execution_time_wrapper import get_execution_time_print

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