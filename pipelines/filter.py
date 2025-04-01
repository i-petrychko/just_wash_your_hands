import os
import sys
from omegaconf import OmegaConf
from typing import List
import argparse

sys.path.append(".")


from schemas.labels import ImageLabelSchema
from schemas.preprocessing import Config
from common.utils import (
    save_unserializable_json,
    read_json,
    sample_truncated_normal,
)


def filter_by_status(
    image_labels: List[ImageLabelSchema], statuses: List
) -> List[ImageLabelSchema]:

    resulted_image_labels = []

    for item in image_labels:

        if item.status not in statuses:
            continue

        resulted_image_labels.append(item)

    return resulted_image_labels


def filter_by_categories(
    image_labels: List[ImageLabelSchema], categories: List
) -> List[ImageLabelSchema]:
    resulted_image_labels = []
    for item in image_labels:

        resulted_labels = []

        for label in item.labels:

            if label.object.name not in categories:
                continue

            resulted_labels.append(label)

        if resulted_labels:
            item.labels = resulted_labels
            resulted_image_labels.append(item)

    return resulted_image_labels


def get_scaled_labels(image_labels: List[ImageLabelSchema], pixel_size: float):

    for image_label in image_labels:

        pixel_width = (
            image_label.labels[0].yolo_annotation.width
            * image_label.labels[0].image_shape.width
        )
        pixel_height = (
            image_label.labels[0].yolo_annotation.height
            * image_label.labels[0].image_shape.height
        )
        microns_width = sample_truncated_normal(
            image_label.labels[0].object.width.min_value,
            image_label.labels[0].object.width.max_value,
            4,
        )[0]
        microns_height = sample_truncated_normal(
            image_label.labels[0].object.height.min_value,
            image_label.labels[0].object.height.max_value,
            4,
        )[0]

        pixel_area = pixel_width * pixel_height
        microns_area = microns_width * microns_height

        actual_pixel_size = (microns_area / pixel_area) ** 0.5

        scaling_cf = pixel_size / actual_pixel_size

        image_label.scaling_cf = scaling_cf

    return image_labels


def main(args):

    config = OmegaConf.load(args.config_path)

    config = OmegaConf.to_container(config, resolve=True)
    config = Config(**config)

    output_dir = config.paths.out_path
    os.makedirs(f"{output_dir}/filtered", exist_ok=True)
    # train set
    labels_json = read_json(f"{config.paths.out_path}/splits/train.json")
    labels = [ImageLabelSchema.from_dict(label_json) for label_json in labels_json]
    labels = filter_by_status(labels, config.filtering.label_statuses)
    labels = filter_by_categories(
        labels, [category.name for category in config.filtering.categories]
    )

    if config.preprocessing is not None and config.preprocessing.pixel_size is not None:
        labels = get_scaled_labels(labels, config.preprocessing.pixel_size)

    save_unserializable_json(labels, f"{output_dir}/filtered/train.json")

    # val set
    labels_json = read_json(f"{config.paths.out_path}/splits/val.json")
    labels = [ImageLabelSchema.from_dict(label_json) for label_json in labels_json]
    labels = filter_by_status(labels, config.filtering.label_statuses)
    labels = filter_by_categories(
        labels, [category.name for category in config.filtering.categories]
    )

    if config.preprocessing is not None and config.preprocessing.pixel_size is not None:
        labels = get_scaled_labels(labels, config.preprocessing.pixel_size)

    save_unserializable_json(labels, f"{output_dir}/filtered/val.json")

    # test set
    labels_json = read_json(f"{config.paths.out_path}/splits/test.json")
    labels = [ImageLabelSchema.from_dict(label_json) for label_json in labels_json]
    labels = filter_by_status(labels, config.filtering.label_statuses)
    labels = filter_by_categories(
        labels, [category.name for category in config.filtering.categories]
    )

    if config.preprocessing is not None and config.preprocessing.pixel_size is not None:
        labels = get_scaled_labels(labels, config.preprocessing.pixel_size)

    save_unserializable_json(labels, f"{output_dir}/filtered/test.json")



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, default="config.yaml")
    args = parser.parse_args()

    main(args)
