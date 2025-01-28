from omegaconf import OmegaConf
import cv2
from typing import List, Tuple
import numpy as np
import copy
import os

import sys

sys.path.append(".")

from preprocessing.schemas import ImageLabelSchema, LabelCoordinatesSchema
from schemas.recipes.preprocessing import Config, DatasetSubsetConfig
from common.utils import save_unserializable_json, read_json
from common.visualizations import get_image_with_targets


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


def scale_image(
    image_label: ImageLabelSchema, image: np.ndarray, scaling_cf: float
) -> (ImageLabelSchema, np.ndarray):
    scaled_image = cv2.resize(
        image,
        dsize=(0, 0),
        fx=1 / scaling_cf,
        fy=1 / scaling_cf,
        interpolation=(cv2.INTER_LANCZOS4 if 1 / scaling_cf > 1 else cv2.INTER_AREA),
    )
    for label in image_label.labels:
        label.coordinates = label.coordinates * (1 / scaling_cf)

    return image_label, scaled_image


def pad_image(
    image_label: ImageLabelSchema, image: np.ndarray, out_dim: Tuple[int, int]
) -> (ImageLabelSchema, np.ndarray):
    height, width, channels = image.shape

    out_height, out_width = out_dim

    top = (out_height - height) // 2 if height < out_height else 0
    bottom = out_height - height - top if height < out_height else 0
    left = (out_width - width) // 2 if width < out_width else 0
    right = out_width - width - left if width < out_width else 0

    padded_image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    for label in image_label.labels:
        label.coordinates = LabelCoordinatesSchema(
            min_x=label.coordinates.min_x + left,
            max_x=label.coordinates.max_x + left,
            min_y=label.coordinates.min_y + top,
            max_y=label.coordinates.max_y + top,
        )

        label.update_yolo_and_area()

    return image_label, padded_image


def crop_image(
    image_label: ImageLabelSchema, image: np.ndarray, out_dim: Tuple[int, int]
) -> (ImageLabelSchema, np.ndarray):

    x_min = min([label.coordinates.min_x for label in image_label.labels])
    y_min = min([label.coordinates.min_y for label in image_label.labels])
    x_max = max([label.coordinates.max_x for label in image_label.labels])
    y_max = max([label.coordinates.max_y for label in image_label.labels])

    x_mid = (x_min + x_max) // 2
    y_mid = (y_min + y_max) // 2

    out_height, out_width = out_dim
    height, width, channels = image.shape

    crop_x_start = max(0, min(x_mid - out_width // 2, width - out_width))
    crop_y_start = max(0, min(y_mid - out_height // 2, height - out_height))
    crop_x_end = crop_x_start + out_width
    crop_y_end = crop_y_start + out_height

    cropped_image = image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    # Adjust bounding box coordinates based on the cropping and placement
    for label in image_label.labels:
        label.coordinates = LabelCoordinatesSchema(
            min_x=label.coordinates.min_x - crop_x_start,
            max_x=label.coordinates.max_x - crop_x_start,
            min_y=label.coordinates.min_y - crop_y_start,
            max_y=label.coordinates.max_y - crop_y_start,
        )

        label.update_yolo_and_area()

    return image_label, cropped_image


def preprocess_image(
    image_label: ImageLabelSchema, config: Config
) -> (ImageLabelSchema, np.ndarray):

    image_label = copy.deepcopy(image_label)
    scaling_cf = image_label.scaling_cf

    image = cv2.imread(image_label.img_path)
    if image is None:
        print(f"Image at {image_label.img_path} could not be loaded.")
        return

    if config.preprocessing.out_channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if scaling_cf is not None:
        image_label, image = scale_image(image_label, image, scaling_cf)
    image_label, image = pad_image(image_label, image, config.preprocessing.out_dim)
    image_label, image = crop_image(image_label, image, config.preprocessing.out_dim)

    return image_label, image


def preprocess_and_save_images(
    image_labels: List[ImageLabelSchema],
    config: Config,
    set_name: str,
    save_targets=False,
) -> List[ImageLabelSchema]:

    preprocessed_image_labels = []
    dest_images_dir_path = f"{config.paths.out_path}/yolo_dataset/images/{set_name}"
    os.makedirs(dest_images_dir_path, exist_ok=True)

    for image_label in image_labels:

        img_name = os.path.basename(image_label.img_path)
        image_label, image = preprocess_image(image_label, config)

        dest_image_path = f"{dest_images_dir_path}/{img_name}"
        cv2.imwrite(dest_image_path, image)

        image_label.img_path = dest_image_path

        if save_targets:
            target_image = get_image_with_targets(image_label)
            target_image_save_path = dest_image_path.replace("images", "targets")
            os.makedirs(os.path.dirname(target_image_save_path), exist_ok=True)
            cv2.imwrite(target_image_save_path, target_image)

        preprocessed_image_labels.append(image_label)

    return preprocessed_image_labels


def main(config: Config):

    test_image_labels_json = read_json(f"{config.paths.out_path}/test.json")
    val_image_labels_json = read_json(f"{config.paths.out_path}/val.json")
    train_image_labels_json = read_json(f"{config.paths.out_path}/train.json")

    test_image_labels = [
        ImageLabelSchema.from_dict(test_image_label_json)
        for test_image_label_json in test_image_labels_json
    ]
    val_image_labels = [
        ImageLabelSchema.from_dict(val_image_label_json)
        for val_image_label_json in val_image_labels_json
    ]
    train_image_labels = [
        ImageLabelSchema.from_dict(train_image_label_json)
        for train_image_label_json in train_image_labels_json
    ]

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

    preprocessed_test_image_labels = preprocess_and_save_images(
        filtered_test_labels, config, "test", True
    )
    preprocessed_val_image_labels = preprocess_and_save_images(
        filtered_val_labels, config, "val", True
    )
    preprocessed_train_image_labels = preprocess_and_save_images(
        filtered_train_labels, config, "train", True
    )


if __name__ == "__main__":
    preprocessing_config_path = "src/recipes/preprocessing/default.yaml"
    config = OmegaConf.load(preprocessing_config_path)

    config = OmegaConf.to_container(config, resolve=True)
    config = Config(**config)

    main(config)
