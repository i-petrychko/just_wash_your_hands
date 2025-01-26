import os
import sys
from omegaconf import OmegaConf
from typing import List

sys.path.append(".")


from preprocessing.schemas import ImageLabelSchema
from schemas.recipes.preprocessing import Config
from common.utils import get_latest_labels_json, save_unserializable_json


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


def main(config: Config):

    labels_json = get_latest_labels_json()
    labels = [ImageLabelSchema.from_dict(label_json) for label_json in labels_json]
    output_dir = config.paths.out_path
    os.makedirs(output_dir, exist_ok=True)

    labels = filter_by_status(labels, config.filtering.label_statuses)

    labels = filter_by_categories(
        labels, [category.name for category in config.filtering.categories]
    )

    save_unserializable_json(labels, f"{output_dir}/filtered.json")


if __name__ == "__main__":

    preprocessing_config_path = "src/recipes/preprocessing/default.yaml"
    config = OmegaConf.load(preprocessing_config_path)

    config = OmegaConf.to_container(config, resolve=True)
    config = Config(**config)

    main(config)
