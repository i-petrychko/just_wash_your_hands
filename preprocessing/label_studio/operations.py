import sys
import os
from typing import List
import copy
import cv2

sys.path.append(".")

from preprocessing.schemas import ImageLabelSchema
from preprocessing.settings import settings
from common.utils import (
    read_json,
    save_unserializable_json,
    get_latest_labels_json,
    get_latest_labels_version,
    get_target_img_path,
)
from common.visualizations import get_image_with_targets
from preprocessing.schemas import (
    Label,
    FilteredOutput,
    FilteredLabel,
    Result,
    Choice,
    Status,
)


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


def filter_filtered_labels(
    filtered_labels: List[FilteredLabel], num_results: int, parasites: List
):

    resulted_filtered_labels = []
    for filtered_label in filtered_labels:
        if len(filtered_label.results) != num_results:
            continue
        for result in filtered_label.results:
            if result.value.rectanglelabels[0] in parasites:
                resulted_filtered_labels.append(filtered_label)
                break

    return resulted_filtered_labels


def filter_filtering_outputs(
    filtering_outputs: List[FilteredOutput], num_results: int, parasites: List
):

    resulted_filtering_outputs = []
    for filtering_output in filtering_outputs:
        if len(filtering_output.annotations[0].result) != num_results:
            continue
        for result in filtering_output.annotations[0].result:
            if (
                isinstance(result, Result)
                and result.value.rectanglelabels[0] in parasites
            ):
                resulted_filtering_outputs.append(filtering_output)
                break

    return resulted_filtering_outputs


def approve_label(filtered_label: FilteredLabel) -> ImageLabelSchema:

    new_label = ImageLabelSchema.from_filtering_results(filtered_label)
    target_img_path = get_target_img_path(filtered_label.img_path)
    target_image = get_image_with_targets(new_label)
    cv2.imwrite(target_img_path, target_image)

    return new_label


def reject_label(current_label: ImageLabelSchema) -> ImageLabelSchema:

    current_label.status = Status.REJECTED
    return current_label


def apply_actions(
    filtered_labels: List[FilteredLabel], current_labels: List[ImageLabelSchema]
):

    current_labels_dict = {
        current_label.img_path: current_label for current_label in current_labels
    }

    for filtered_label in filtered_labels:

        img_path = filtered_label.img_path
        current_label = copy.deepcopy(current_labels_dict[img_path])

        if filtered_label.choice == Choice.APPROVE:

            new_label = approve_label(filtered_label)
            current_labels_dict[img_path] = new_label

        elif filtered_label.choice == Choice.REJECT:

            new_label = reject_label(current_label)
            current_labels_dict[img_path] = new_label

        else:
            print("Unknown action")

    new_labels = list(current_labels_dict.values())

    return new_labels


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

    directory, filename = os.path.split(filtering_results_file_path)
    new_filename = f"processed_{filename}"
    destination_path = os.path.join(directory, new_filename)
    save_unserializable_json(filtered_labels, destination_path)

    current_labels = [
        ImageLabelSchema.from_dict(image_label)
        for image_label in get_latest_labels_json()
    ]
    current_version = get_latest_labels_version()
    new_labels = apply_actions(filtered_labels, current_labels)
    save_unserializable_json(
        new_labels,
        f"{settings.data_path}/labels/labels_version_{current_version+1}.json",
    )


if __name__ == "__main__":
    filter_dataset("preprocessing/label_studio/results/Paragonimus_spp_all.json")
    convert_image_labels_to_label_studio()
