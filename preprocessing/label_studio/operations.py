import sys

sys.path.append(".")

from preprocessing.schemas import ImageLabelSchema
from preprocessing.settings import settings
from common.utils import (
    read_json,
    save_unserializable_json,
    get_latest_labels_json,
    get_latest_labels_version,
)
from preprocessing.label_studio.schemas import Label, FilteredOutput, FilteredLabel


def convert_image_labels_to_label_studio():
    labels = get_latest_labels_json()

    image_labels = [ImageLabelSchema.from_dict(label) for label in labels]

    label_studio_labels = [
        Label.from_image_label(image_label) for image_label in image_labels
    ]

    latest_labels_version = get_latest_labels_version()

    save_unserializable_json(
        label_studio_labels,
        f"{settings.data_path}/labels/label_studio_labels_version_{latest_labels_version}.json",
    )


def filter_dataset(filtering_results_file_path):

    filtering_outputs_json = read_json(filtering_results_file_path)
    filtering_outputs = [
        FilteredOutput.from_dict(filtering_output)
        for filtering_output in filtering_outputs_json
    ]
    filtered_labels = [
        FilteredLabel.from_filtered_output(filtered_output)
        for filtered_output in filtering_outputs
    ]
    save_unserializable_json(filtered_labels, "1.json")


if __name__ == "__main__":
    filter_dataset("preprocessing/label_studio/results/Paragonimus_spp_all.json")
