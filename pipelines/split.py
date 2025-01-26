from sklearn.model_selection import train_test_split
from typing import List, Tuple, Any
from omegaconf import OmegaConf

import sys

sys.path.append(".")


from preprocessing.schemas import ImageLabelSchema
from schemas.recipes.preprocessing import Config
from common.utils import save_unserializable_json, read_json


def stratified_split(
    data: List[Any],
    labels: List[Any],
    train_size: float = 0.8,
    val_size: float = 0.0,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Perform a stratified split of data into train, validation, and test sets.

    Args:
        data (List[Any]): The list of data objects.
        labels (List[Any]): The list of labels corresponding to the data.
        train_size (float): Proportion of the data to include in the train set.
        val_size (float): Proportion of the data to include in the validation set.
        test_size (float): Proportion of the data to include in the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[List[Any], List[Any], List[Any]]: Train, validation, and test splits.
    """
    # Calculate proportions for splitting train and test first
    train_ratio = train_size / (train_size + test_size)

    # Step 1: Split into train+val and test sets
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, stratify=labels, random_state=random_state
    )

    if val_size > 0:
        # Step 2: Split train+val into train and val sets
        val_ratio = val_size / (train_size + val_size)
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_val_data,
            train_val_labels,
            test_size=val_ratio,
            stratify=train_val_labels,
            random_state=random_state,
        )
    else:
        # If no validation set is required
        train_data, val_data = train_val_data, []
        train_labels, val_labels = train_val_labels, []

    return train_data, val_data, test_data


def main(config):

    labels_json = read_json(f"{config.paths.out_path}/filtered.json")
    data = [ImageLabelSchema.from_dict(label_json) for label_json in labels_json]
    labels = [label.labels[0].object.name for label in data]
    train_data, val_data, test_data = [], [], []

    if config.split.type == "stratified":
        train_data, val_data, test_data = stratified_split(
            data,
            labels,
            config.split.ratio[0],
            config.split.ratio[1],
            config.split.ratio[2],
            config.split.seed,
        )
    else:
        print("unknown split type")

    save_unserializable_json(train_data, f"{config.paths.out_path}/train.json")
    save_unserializable_json(val_data, f"{config.paths.out_path}/val.json")
    save_unserializable_json(test_data, f"{config.paths.out_path}/test.json")


if __name__ == "__main__":
    preprocessing_config_path = "src/recipes/preprocessing/default.yaml"
    config = OmegaConf.load(preprocessing_config_path)

    config = OmegaConf.to_container(config, resolve=True)
    config = Config(**config)

    main(config)
