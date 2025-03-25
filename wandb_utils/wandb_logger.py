import wandb
import torch
from PIL import Image
import numpy as np
import os

class WandBLogger:
    def __init__(self, name=None, project_name=None, config=None, entity=None):
        """Initialize WandB logger
        Args:
            project_name (str): Name of the project
            config (dict): Configuration parameters
            entity (str): WandB username or team name
        """
        self.run = wandb.init(name=name, project=project_name, config=config, entity=entity)
        
    def log_metrics(self, metrics, step=None):
        """Log training/validation metrics
        Args:
            metrics (dict): Dictionary of metrics to log
            step (int): Current step/epoch
        """
        wandb.log(metrics, step=step)
    
    def log_model_predictions(self, images, pred_boxes, pred_labels, 
                            pred_scores, gt_boxes=None, gt_labels=None,
                            score_threshold=0.5):
        """Log images with predictions and ground truth boxes
        Args:
            images: List of images
            pred_boxes: List of predicted bounding boxes
            pred_labels: List of predicted labels
            pred_scores: List of prediction scores
            gt_boxes: List of ground truth boxes (optional)
            gt_labels: List of ground truth labels (optional)
            score_threshold: Threshold for showing predictions
        """
        for idx, (img, boxes, labels, scores) in enumerate(
            zip(images, pred_boxes, pred_labels, pred_scores)):
            
            # Convert tensor to PIL Image if needed
            if isinstance(img, torch.Tensor):
                img = torch.clamp(img * 255, 0, 255).byte().permute(1, 2, 0).cpu().numpy()
                img = Image.fromarray(img)
            
            # Filter predictions by score threshold
            mask = scores > score_threshold
            boxes = boxes[mask]
            labels = labels[mask]
            scores = scores[mask]
            
            boxes_data = {
                "predictions": {
                    "box_data": [
                        {
                            "position": {
                                "minX": float(box[0]),
                                "minY": float(box[1]),
                                "maxX": float(box[2]),
                                "maxY": float(box[3])
                            },
                            "class_id": int(label),
                            "box_caption": f"Pred: {int(label)} ({score:.2f})",
                            "scores": {"confidence": float(score)}
                        }
                        for box, label, score in zip(boxes, labels, scores)
                    ]
                }
            }
            
            # Add ground truth boxes if available
            if gt_boxes is not None and gt_labels is not None:
                boxes_data["ground_truth"] = {
                    "box_data": [
                        {
                            "position": {
                                "minX": float(box[0]),
                                "minY": float(box[1]),
                                "maxX": float(box[2]),
                                "maxY": float(box[3])
                            },
                            "class_id": int(label),
                            "box_caption": f"GT: {int(label)}"
                        }
                        for box, label in zip(gt_boxes[idx], gt_labels[idx])
                    ]
                }
            
            wandb.log({"predictions": wandb.Image(img, boxes=boxes_data)})
    
    def upload_checkpoint(self, checkpoint_path):
        """Save model checkpoint file to wandb run directory
        Args:
            checkpoint_path (str): Path to checkpoint file 
            filename (str): Name of the checkpoint file (optional)
        """
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint path {checkpoint_path} does not exist")
            return
            
        # Upload checkpoint directly to wandb
        self.run.save(
            str(checkpoint_path)
        )

    def upload_file(self, file_path, base_path=None):
        """Upload file to wandb run directory
        Args:
            file_path (str): Path to file 
        """
        if not os.path.exists(file_path):
            print(f"Warning: File path {file_path} does not exist")
            return
            
        # Upload checkpoint directly to wandb
        self.run.save(
            str(file_path),
            base_path=base_path
        )
    
    
    def log_bad_predictions(self, images, pred_boxes, pred_labels, pred_scores,
                          gt_boxes, gt_labels, iou_threshold=0.5):
        """Log images where predictions significantly differ from ground truth
        Args:
            images: List of images
            pred_boxes: Predicted boxes
            pred_labels: Predicted labels
            pred_scores: Prediction scores
            gt_boxes: Ground truth boxes
            gt_labels: Ground truth labels
            iou_threshold: IoU threshold for considering a prediction correct
        """
        def calculate_iou(box1, box2):
            # Calculate intersection over union between two boxes
            x1, y1, x2, y2 = box1
            x3, y3, x4, y4 = box2
            
            xi1 = max(x1, x3)
            yi1 = max(y1, y3)
            xi2 = min(x2, x4)
            yi2 = min(y2, y4)
            
            inter_width = max(0, xi2 - xi1)
            inter_height = max(0, yi2 - yi1)
            inter_area = inter_width * inter_height
            
            box1_area = (x2 - x1) * (y2 - y1)
            box2_area = (x4 - x3) * (y4 - y3)
            union_area = box1_area + box2_area - inter_area
            
            iou = inter_area / union_area if union_area > 0 else 0
            return iou
        
        for idx, (img, p_boxes, p_labels, p_scores, g_boxes, g_labels) in enumerate(
            zip(images, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)):
            
            # Check if prediction significantly differs from ground truth
            max_ious = []
            for p_box, p_label in zip(p_boxes, p_labels):
                ious = [calculate_iou(p_box, g_box) for g_box in g_boxes]
                max_iou = max(ious) if ious else 0
                max_ious.append(max_iou)
            
            # If any prediction has low IoU with all ground truth boxes
            if any(iou < iou_threshold for iou in max_ious):
                self.log_model_predictions(
                    [img], [p_boxes], [p_labels], [p_scores],
                    [g_boxes], [g_labels]
                )
    
    def finish(self):
        """Close wandb run"""
        wandb.finish() 