from typing import List, Dict

import os 
import sys
import uuid
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from schemas.labels import ImageLabelSchema, LabelSchema, ObjectSchema, LabelCoordinatesSchema, ImageShape
from common.utils import read_json, save_unserializable_json
def convert_coco_to_image_labels(coco_json: Dict, img_dir: str) -> List[ImageLabelSchema]:
    image_id_to_filename = {img["id"]: img["file_name"] for img in coco_json["images"]}
    image_id_to_size = {img["id"]: (img["width"], img["height"]) for img in coco_json["images"]}
    
    image_labels = []

    annotations_by_image = {}
    for ann in coco_json["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        
        category_id = ann["category_id"]
        category_name = next((cat["name"] for cat in coco_json["categories"] if cat["id"] == category_id), f"Unknown_{category_id}")

        x, y, w, h = ann["bbox"]
        width, height = image_id_to_size[img_id]

        label_schema = LabelSchema(
            object=ObjectSchema.from_initial_name(category_name),
            coordinates=LabelCoordinatesSchema.from_xywh(x, y, w, h),
            image_shape=ImageShape(width=width, height=height)
        )

        annotations_by_image[img_id].append(label_schema)

    for img_id, labels in annotations_by_image.items():
        image_labels.append(
            ImageLabelSchema(
                img_path=os.path.join(img_dir, image_id_to_filename[img_id]),
                labels=labels
            )
        )

    return image_labels

def convert_image_labels_to_coco(image_labels: List[ImageLabelSchema]) -> Dict:
    coco_json = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_id_map = {}  # Map img_path to unique integer IDs
    category_id_map = {}  # Map category names to unique integer IDs
    annotation_id = 1  # Unique counter for annotations
    image_id = 1  # Unique counter for images

    for image_label in image_labels:
        # Ensure unique image ID
        if image_label.img_path not in image_id_map:
            image_id_map[image_label.img_path] = image_id
            coco_json["images"].append({
                "id": image_id,
                "file_name": image_label.img_path,
                "width": image_label.labels[0].image_shape.width,
                "height": image_label.labels[0].image_shape.height    
            })
            image_id += 1

        img_id = image_id_map[image_label.img_path]

        for label in image_label.labels:
            # Ensure unique category ID
            category_name = label.object.name
            if category_name not in category_id_map:
                coco_json["categories"].append({
                    "id": label.object.category_id,
                    "name": category_name
                })
            
            bbox = label.get_xywh()
            coco_json["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": label.object.category_id,
                "bbox": bbox,
                "iscrowd": 0,
                "area": bbox[-1] * bbox[-2]
            })
            annotation_id += 1

    return coco_json


if __name__ == "__main__":
    coco_json = read_json("data/icip/annotations/val.json")
    image_labels = convert_coco_to_image_labels(coco_json, "data/icip/val")
    save_unserializable_json(image_labels, "val.json")