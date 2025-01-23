import sys

sys.path.append(".")

from preprocessing.schemas import ImageLabelSchema
from preprocessing.settings import settings
from common.utils import read_json, save_unserializable_json
from preprocessing.label_studio.schemas import Label


def convert_image_labels_to_label_studio():
    labels = read_json(f"{settings.data_path}/labels.json")

    image_labels = [ImageLabelSchema.from_dict(label) for label in labels]

    label_studio_labels = [
        Label.from_image_label(image_label) for image_label in image_labels
    ]

    save_unserializable_json(
        label_studio_labels, f"{settings.data_path}/label_studio_labels.json"
    )

def filter_dataset():
    pass


if __name__ == "__main__":
    convert_image_labels_to_label_studio()
