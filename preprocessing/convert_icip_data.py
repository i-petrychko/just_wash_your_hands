import os
import sys
import cv2

sys.path.append(".")

from preprocessing.settings import settings
from common.utils import read_json, save_unserializable_json
from common.visualizations import get_image_with_targets
from preprocessing.schemas import (
    ImageLabelSchema,
    LabelSchema,
    ObjectSchema,
    LabelCoordinatesSchema,
    ImageShape,
)


def convert_icip_labels_to_image_label_schema(labels, root_dir_path):
    image_id_to_annotation = {}
    category_id_to_category_name = {
        category["id"]: category["name"] for category in labels["categories"]
    }

    for image_data in labels["images"]:
        image_id = image_data["id"]
        image_filename = image_data["file_name"]
        height = image_data["height"]
        width = image_data["width"]
        image_id_to_annotation[image_id] = {
            "image_filename": image_filename,
            "height": height,
            "width": width,
            "annotation": [],
        }

    for annotation in labels["annotations"]:
        image_id = annotation["image_id"]
        annotation["name"] = category_id_to_category_name[annotation["category_id"]]
        image_id_to_annotation[image_id]["annotation"].append(annotation)

    labels = list(image_id_to_annotation.values())

    image_labels = [
        ImageLabelSchema(
            img_path=f"{root_dir_path}/{label['image_filename']}",
            labels=[
                LabelSchema(
                    object=ObjectSchema.from_initial_name(annotation["name"]),
                    coordinates=LabelCoordinatesSchema.from_xywh(*annotation["bbox"]),
                    image_shape=ImageShape(
                        width=label["width"], height=label["height"]
                    ),
                )
                for annotation in label["annotation"]
            ],
        )
        for label in labels
    ]

    return image_labels


def main():
    merged_image_labels = []

    train_labels = read_json(settings.orig_icip_train_labels_file_path)
    train_image_labels = convert_icip_labels_to_image_label_schema(
        train_labels, settings.orig_icip_train_images_dir_path
    )
    merged_image_labels.extend(train_image_labels)

    test_labels = read_json(settings.orig_icip_test_labels_file_path)
    test_image_labels = convert_icip_labels_to_image_label_schema(
        test_labels, settings.orig_icip_test_images_dir_path
    )
    merged_image_labels.extend(test_image_labels)

    images_dataset_dir_path = f"{settings.data_path}/images"
    targets_dataset_dir_path = f"{settings.data_path}/targets"

    os.makedirs(images_dataset_dir_path, exist_ok=True)
    os.makedirs(targets_dataset_dir_path, exist_ok=True)

    for image_idx, image_label in enumerate(merged_image_labels):
        orig_img_path = image_label.img_path
        orig_image_save_path = f"{images_dataset_dir_path}/{image_idx}.jpg"
        target_image_save_path = f"{targets_dataset_dir_path}/{image_idx}.jpg"

        image_label.img_path = orig_image_save_path

        if os.path.exists(orig_image_save_path):
            continue

        orig_image = cv2.imread(orig_img_path)
        target_image = get_image_with_targets(image_label)

        cv2.imwrite(orig_image_save_path, orig_image)
        cv2.imwrite(target_image_save_path, target_image)

    save_unserializable_json(merged_image_labels, f"{settings.data_path}/labels.json")


if __name__ == "__main__":
    main()
