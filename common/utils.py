import json
import sys
import os
import numpy as np
from scipy.stats import truncnorm

sys.path.append(".")

from preprocessing.settings import settings


def get_latest_labels_version():

    labels_dir_path = f"{settings.data_path}/labels"
    labels_files = list(
        filter(
            lambda filename: filename.startswith("labels_version")
            and filename.endswith(".json"),
            os.listdir(labels_dir_path),
        )
    )
    labels_files.sort(key=lambda filename: int(filename.split("_")[-1].split(".")[0]))

    return int(labels_files[-1].split("_")[-1].split(".")[0])


def get_target_img_path(img_path):
    return img_path.replace("images", "targets")


def sample_truncated_normal(
    min_val: float, max_val: float, std_dev: float, size: int = 1
):
    """
    Sample values from a truncated normal distribution.

    Args:
        min_val (float): The minimum value of the distribution.
        max_val (float): The maximum value of the distribution.
        mean (float): The mean of the normal distribution.
        std_dev (float): The standard deviation of the normal distribution.
        size (int): Number of samples to generate.

    Returns:
        np.ndarray: Array of sampled values.
    """
    # Compute the lower and upper bounds in standard normal units
    mean = max_val - min_val
    lower_bound = (min_val - mean) / std_dev
    upper_bound = (max_val - mean) / std_dev

    # Generate samples from the truncated normal distribution
    samples = truncnorm.rvs(
        lower_bound, upper_bound, loc=mean, scale=std_dev, size=size
    )
    return samples


def get_latest_labels_json():

    labels_dir_path = f"{settings.data_path}/labels"
    labels_files = list(
        filter(
            lambda filename: filename.startswith("labels_version")
            and filename.endswith(".json"),
            os.listdir(labels_dir_path),
        )
    )
    labels_files.sort(key=lambda filename: int(filename.split("_")[-1].split(".")[0]))

    return read_json(f"{labels_dir_path}/{labels_files[-1]}")


def get_latest_label_studio_labels_json():

    labels_dir_path = f"{settings.data_path}/labels"
    labels_files = list(
        filter(
            lambda filename: filename.startswith("label_studio_labels_version")
            and filename.endswith(".json"),
            os.listdir(labels_dir_path),
        )
    )
    labels_files.sort(key=lambda filename: int(filename.split("_")[-1].split(".")[0]))

    return read_json(f"{labels_dir_path}/{labels_files[-1]}")


def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_json(data, path):
    with open(path, "w") as file:
        json.dump(data, file, indent=4)


def save_unserializable_json(data, path):
    """
    Save a list of Pydantic models to a JSON file.
    """
    with open(path, "w") as file:
        json.dump([item.model_dump() for item in data], file, indent=4)


if __name__ == "__main__":
    print(get_latest_labels_json())
    print(get_latest_label_studio_labels_json())
