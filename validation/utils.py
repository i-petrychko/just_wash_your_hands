import os
import sys
import json
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.base_model import ImagePrediction
from validation.constants import NO_PARASITE_CLASS


def extract_image_paths(dataset_labels_path: str, imgs_dir: str) -> List[str]:
    # dataset_labels_path is a json file with labels in coco format
    with open(dataset_labels_path, "r") as f:
        data = json.load(f)
    return [
        f"{imgs_dir}/{os.path.join(data['images'][i]['file_name'])}"
        for i in range(len(data["images"]))
    ]


def convert_coco_to_image_predictions(data: dict, imgs_dir: str) -> List[ImagePrediction]:
    # Create a mapping from image_id to ImagePrediction
    image_predictions = {}

    # Initialize predictions for each image
    for image in data["images"]:
        image_predictions[image["id"]] = ImagePrediction(
            labels=[],
            normalized_cxcywhs=[],
            confidences=[],  # For ground truth, all confidences will be 1.0
            img_path=f"{imgs_dir}/{image['file_name']}",
        )

    # Process annotations
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        # Get image width and height for normalization
        img_info = next(img for img in data["images"] if img["id"] == image_id)
        img_width = img_info["width"]
        img_height = img_info["height"]

        # Get bbox in [x, y, width, height] format and normalize
        x, y, w, h = ann["bbox"]

        # Convert to center coordinates and normalize
        cx = (x + w / 2) / img_width
        cy = (y + h / 2) / img_height
        norm_w = w / img_width
        norm_h = h / img_height

        # Add to image predictions
        image_predictions[image_id].labels.append(ann["category_id"])
        image_predictions[image_id].normalized_cxcywhs.append((cx, cy, norm_w, norm_h))
        image_predictions[image_id].confidences.append(1.0)

    return list(image_predictions.values())


def calculate_iou(box1: Tuple, box2: Tuple) -> float:
    # Convert from cxcywh to x1y1x2y2
    cx1, cy1, w1, h1 = box1
    cx2, cy2, w2, h2 = box2

    x11 = cx1 - w1 / 2
    y11 = cy1 - h1 / 2
    x12 = cx1 + w1 / 2
    y12 = cy1 + h1 / 2

    x21 = cx2 - w2 / 2
    y21 = cy2 - h2 / 2
    x22 = cx2 + w2 / 2
    y22 = cy2 + h2 / 2

    # Calculate intersection coordinates
    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def match_predictions(
    gts: List[ImagePrediction],
    preds: List[ImagePrediction],
    iou_threshold: float,
    min_confidence: float,
) -> Tuple[List[int], List[int], List[float]]:
    """Match predictions to ground truth boxes"""
    y_true = []
    y_pred = []
    y_scores = []

    img_path_to_gt = {gt.img_path: gt for gt in gts}
    matched_gt_imgs = set()

    for pred in preds:
        gt = img_path_to_gt[pred.img_path]
        matched_gt_imgs.add(pred.img_path)
        matched_gt_idxs = set()

        for pred_idx, pred_bbox in enumerate(pred.normalized_cxcywhs):
            if pred.confidences[pred_idx] < min_confidence:
                continue

            best_iou = 0
            best_iou_gt_idx = -1

            for gt_idx, gt_bbox in enumerate(gt.normalized_cxcywhs):
                if gt_idx in matched_gt_idxs:
                    continue
                iou = calculate_iou(gt_bbox, pred_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_iou_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                y_true.append(gt.labels[best_iou_gt_idx])
                y_pred.append(pred.labels[pred_idx])
                y_scores.append(pred.confidences[pred_idx])
                matched_gt_idxs.add(best_iou_gt_idx)
            else:
                y_true.append(NO_PARASITE_CLASS)
                y_pred.append(pred.labels[pred_idx])
                y_scores.append(pred.confidences[pred_idx])

        for gt_idx, gt_label in enumerate(gt.labels):
            if gt_idx not in matched_gt_idxs:
                y_true.append(gt_label)
                y_pred.append(NO_PARASITE_CLASS)
                y_scores.append(0)

    for gt in gts:
        if gt.img_path not in matched_gt_imgs:
            for gt_idx, gt_label in enumerate(gt.labels):
                y_true.append(gt_label)
                y_pred.append(NO_PARASITE_CLASS)
                y_scores.append(0)

    return y_true, y_pred, y_scores
