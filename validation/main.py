import argparse
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.rt_detr_model import RTDETRModel
from validation.validation import validate_predictions
from validation.utils import convert_coco_to_image_predictions, extract_image_paths
from models.base_model import ImagePrediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--dataset_labels", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--iou_threshold", type=float, required=True)
    parser.add_argument("--min_confidence", type=float, required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = RTDETRModel(args.model_config, args.model_checkpoint)
    img_paths = extract_image_paths(args.dataset_labels)
    predictions = model.predict(img_paths, args.iou_threshold, args.min_confidence)
    # save predictions to output directory
    with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
        json.dump([prediction.model_dump() for prediction in predictions], f)
    # run validation
    # load ground truth labels
    with open(args.dataset_labels, "r") as f:
        data = json.load(f)
    # load predictions
    with open(os.path.join(args.output_dir, "predictions.json"), "r") as f:
        predictions = json.load(f)
    predictions = [ImagePrediction.from_dict(prediction) for prediction in predictions]
    gt_labels = convert_coco_to_image_predictions(data)
    print(len(gt_labels), len(predictions))
    # run validation
    validate_predictions(
        gt_labels, predictions, args.iou_threshold, args.min_confidence, args.output_dir
    )


if __name__ == "__main__":
    main()
