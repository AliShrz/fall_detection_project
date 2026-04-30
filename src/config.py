import yaml
import os


def load_config(config_path: str = None) -> dict:
    """
    Load the project config.yaml and return it as a dictionary.

    Parameters
    ----------
    config_path : str, optional
        Path to config.yaml. If not provided, automatically finds it
        relative to this file's location.

    Returns
    -------
    dict : all config values
    """
    if config_path is None:
        # Go up one level from src/ to project root, then into configs/
        src_dir     = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(src_dir)
        config_path = os.path.join(project_dir, "configs", "config.yaml")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


# Load once when the module is imported
# Any file that does `from src.config import CFG` gets this dictionary
CFG = load_config()