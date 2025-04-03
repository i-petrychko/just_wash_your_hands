import argparse
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.rt_detr_model import RTDETRModel
from validation.validation import validate_predictions
from validation.utils import convert_coco_to_image_predictions, extract_image_paths
from models.base_model import ImagePrediction
from wandb_utils.run_operations import get_run_id, download_file, upload_directory
from src.core import YAMLConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--iou_threshold", type=float, required=True)
    parser.add_argument("--min_confidence", type=float, required=True)
    args = parser.parse_args()

    output_dir = "validation_results"

    cfg = YAMLConfig(args.model_config)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/val", exist_ok=True)

    run_id = get_run_id(args.model_config)
    download_dir = f"{run_id}_files"
    os.makedirs(download_dir, exist_ok=True)
    download_file(
        run_id,
        cfg.wandb_project_name,
        cfg.wandb_entity,
        f"{cfg.output_dir}/checkpoint_best.pth",
        download_dir,
    )
    download_file(
        run_id,
        cfg.wandb_project_name,
        cfg.wandb_entity,
        "train.json",
        download_dir,
    )
    download_file(
        run_id, cfg.wandb_project_name, cfg.wandb_entity, "val.json", download_dir
    )

    model = RTDETRModel(args.model_config, f"{download_dir}/{cfg.output_dir}/checkpoint_best.pth")

    # train validation
    img_paths = extract_image_paths(f"{download_dir}/train.json", imgs_dir=cfg.train_dataloader.dataset.img_folder)
    predictions = model.predict(img_paths, args.iou_threshold, args.min_confidence)
    ## save predictions to output directory
    with open(f"{output_dir}/train/predictions.json", "w") as f:
        json.dump([prediction.model_dump() for prediction in predictions], f)
    ## download annotations
    with open(f"{download_dir}/train.json", "r") as f:
        data = json.load(f)

    with open(f"{output_dir}/train/predictions.json", "r") as f:
        predictions = json.load(f)
    ## convert predictions to ImagePrediction objects
    predictions = [ImagePrediction.from_dict(prediction) for prediction in predictions]
    gt_labels = convert_coco_to_image_predictions(data, imgs_dir=cfg.train_dataloader.dataset.img_folder)
    # run validation
    validate_predictions(
        gt_labels,
        predictions,
        args.iou_threshold,
        args.min_confidence,
        f"{output_dir}/train",
    )

    # val validation
    img_paths = extract_image_paths(f"{download_dir}/val.json", imgs_dir=cfg.val_dataloader.dataset.img_folder)
    predictions = model.predict(img_paths, args.iou_threshold, args.min_confidence)
    ## save predictions to output directory
    with open(f"{output_dir}/val/predictions.json", "w") as f:
        json.dump([prediction.model_dump() for prediction in predictions], f)
    ## download annotations
    with open(f"{download_dir}/val.json", "r") as f:
        data = json.load(f)

    with open(f"{output_dir}/val/predictions.json", "r") as f:
        predictions = json.load(f)
    ## convert predictions to ImagePrediction objects
    predictions = [ImagePrediction.from_dict(prediction) for prediction in predictions]
    gt_labels = convert_coco_to_image_predictions(data, imgs_dir=cfg.val_dataloader.dataset.img_folder)
    # run validation
    validate_predictions(
        gt_labels,
        predictions,
        args.iou_threshold,
        args.min_confidence,
        f"{output_dir}/val",
    )

    # test validation
    img_paths = extract_image_paths(f"{download_dir}/test.json", imgs_dir=cfg.test_dataloader.dataset.img_folder)
    predictions = model.predict(img_paths, args.iou_threshold, args.min_confidence)
    ## save predictions to output directory
    with open(f"{output_dir}/test/predictions.json", "w") as f:
        json.dump([prediction.model_dump() for prediction in predictions], f)
    ## download annotations
    with open(f"{download_dir}/test.json", "r") as f:
        data = json.load(f)

    with open(f"{output_dir}/test/predictions.json", "r") as f:
        predictions = json.load(f)
    ## convert predictions to ImagePrediction objects
    predictions = [ImagePrediction.from_dict(prediction) for prediction in predictions]
    gt_labels = convert_coco_to_image_predictions(data, imgs_dir=cfg.test_dataloader.dataset.img_folder)
    # run validation
    validate_predictions(
        gt_labels,
        predictions,
        args.iou_threshold,
        args.min_confidence,
        f"{output_dir}/test",
    )

    upload_directory(run_id, cfg.wandb_project_name, cfg.wandb_entity, output_dir)


if __name__ == "__main__":
    main()