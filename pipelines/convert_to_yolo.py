from omegaconf import OmegaConf
import cv2
from typing import List

import sys

sys.path.append(".")

from preprocessing.schemas import ImageLabelSchema
from schemas.recipes.preprocessing import Config, DatasetSubsetConfig
from common.utils import save_unserializable_json, read_json


def filter_preprocessed_labels_by_scaling_cf(
    labels: List[ImageLabelSchema], category_to_config: dict[str, DatasetSubsetConfig]
):

    resulted_labels = []

    for label in labels:
        name = label.labels[0].object.name
        if (
            category_to_config[name].min_scaling_cf
            <= label.scaling_cf
            <= category_to_config[name].max_scaling_cf
        ):
            resulted_labels.append(label)

    return resulted_labels


def filter_preprocessed_labels_by_relative_area(
    labels: List[ImageLabelSchema], category_to_config: dict[str, DatasetSubsetConfig]
):
    resulted_labels = []

    for label in labels:
        name = label.labels[0].object.name
        min_relative_area = min(img_label.relative_area for img_label in label.labels)
        if (
            category_to_config[name].min_relative_area
            <= min_relative_area
            <= category_to_config[name].max_relative_area
        ):
            resulted_labels.append(label)

    return resulted_labels


def filter_labels(
    labels: List[ImageLabelSchema],
    category_to_config: dict[str, DatasetSubsetConfig],
    skip_scaling_cf: bool,
):

    filtered_labels = filter_preprocessed_labels_by_relative_area(
        labels, category_to_config
    )

    if not skip_scaling_cf:
        filtered_labels = filter_preprocessed_labels_by_scaling_cf(
            filtered_labels, category_to_config
        )

    return filtered_labels


def save_labels_in_yolo_format(label: ImageLabelSchema, save_path: str):
    pass


def preprocess_and_save_images(
    image_labels: List[ImageLabelSchema], config: Config
) -> List[ImageLabelSchema]:

    pass


def main(config: Config):

    test_image_labels = read_json(f"{config.paths.out_path}/test.json")
    val_image_labels = read_json(f"{config.paths.out_path}/val.json")
    train_image_labels = read_json(f"{config.paths.out_path}/train.json")

    if config.preprocessing.pixel_size is not None:
        skip_scaling_cf = False
    else:
        skip_scaling_cf = True

    filtered_test_labels = filter_labels(
        test_image_labels,
        config.filtering.get_set_config_dict("test_set"),
        skip_scaling_cf,
    )
    filtered_val_labels = filter_labels(
        val_image_labels,
        config.filtering.get_set_config_dict("val_set"),
        skip_scaling_cf,
    )
    filtered_train_labels = filter_labels(
        train_image_labels,
        config.filtering.get_set_config_dict("train_set"),
        skip_scaling_cf,
    )


if __name__ == "__main__":
    preprocessing_config_path = "src/recipes/preprocessing/default.yaml"
    config = OmegaConf.load(preprocessing_config_path)

    config = OmegaConf.to_container(config, resolve=True)
    config = Config(**config)

    main(config)
