import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from typing import List
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.base_model import BaseModel, ImagePrediction
from torch.cuda.amp import autocast
from src.core.yaml_config import YAMLConfig
from tqdm import tqdm


class RTDETRModel(BaseModel):
    def __init__(self, config_path: str, checkpoint_path: str) -> None:
        super().__init__()

        # Load config and checkpoint
        self.cfg = YAMLConfig(config_path, resume=checkpoint_path)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]

        # Load state dict and create model
        self.cfg.model.load_state_dict(state)
        self.model = self.cfg.model.deploy()
        self.postprocessor = self.cfg.postprocessor.deploy()

        # Move model to appropriate device
        self.model = self.model.to(self.device)

        # Setup transform
        self.transform = T.Compose(
            [
                T.Resize((640, 640)),
                T.ToTensor(),
            ]
        )

    def predict(
        self,
        paths: List[str],
        iou: float = 0.5,
        min_conf: float = 0.3,
        batch_size: int = 1,
    ) -> List[ImagePrediction]:
        predictions = []

        for path in tqdm(paths, desc="Predicting"):
            # Load and preprocess image
            image = Image.open(path).convert("RGB")
            w, h = image.size
            orig_size = torch.tensor([w, h])[None].to(self.device)
            img_tensor = self.transform(image)[None].to(self.device)

            # Model inference
            with autocast():
                outputs = self.model(img_tensor)
                outputs = self.postprocessor(outputs, orig_size)

            labels, boxes, scores = outputs

            # Convert to numpy for post-processing
            labels = labels.cpu().detach().numpy()
            boxes = boxes.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()

            # Apply NMS and confidence threshold
            labels, boxes, scores = self._postprocess(
                labels[0], boxes[0], scores[0], iou, min_conf
            )

            # Convert boxes to normalized cxcywh format
            normalized_boxes = self._xyxy_to_normalized_cxcywh(boxes[0], w, h)

            predictions.append(
                ImagePrediction(
                    labels=labels[0].tolist(),
                    normalized_cxcywhs=normalized_boxes,
                    confidences=scores[0].tolist(),
                    img_path=path,
                )
            )

        return predictions

    def _postprocess(self, labels, boxes, scores, iou_threshold, conf_threshold):
        # Filter by confidence first
        mask = scores > conf_threshold
        labels = labels[mask]
        boxes = boxes[mask]
        scores = scores[mask]

        def calculate_iou(box1, box2):
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
            iou = inter_area / union_area if union_area != 0 else 0
            return iou

        # Apply NMS
        merged_labels, merged_boxes, merged_scores = [], [], []
        used_indices = set()

        for i in range(len(boxes)):
            if i in used_indices:
                continue

            current_box = boxes[i]
            current_label = labels[i]
            current_score = scores[i]

            boxes_to_merge = [current_box]
            scores_to_merge = [current_score]
            used_indices.add(i)

            for j in range(i + 1, len(boxes)):
                if j in used_indices or labels[j] != current_label:
                    continue

                if calculate_iou(current_box, boxes[j]) >= iou_threshold:
                    boxes_to_merge.append(boxes[j])
                    scores_to_merge.append(scores[j])
                    used_indices.add(j)

            # Merge boxes
            boxes_array = np.array(boxes_to_merge)
            merged_box = [
                np.min(boxes_array[:, 0]),
                np.min(boxes_array[:, 1]),
                np.max(boxes_array[:, 2]),
                np.max(boxes_array[:, 3]),
            ]

            merged_boxes.append(merged_box)
            merged_labels.append(current_label)
            merged_scores.append(max(scores_to_merge))

        return (
            [np.array(merged_labels)],
            [np.array(merged_boxes)],
            [np.array(merged_scores)],
        )

    def _xyxy_to_normalized_cxcywh(self, boxes, img_width, img_height):
        normalized_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box

            # Convert to center coordinates
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            # Normalize
            cx /= img_width
            cy /= img_height
            w /= img_width
            h /= img_height

            normalized_boxes.append((cx, cy, w, h))

        return normalized_boxes


if __name__ == "__main__":
    model = RTDETRModel(
        config_path="recipes/rtdetr_r18vd_6x_icip.yml",
        checkpoint_path="checkpoint_best.pth",
    )
    predictions = model.predict(
        [
            "data/icip/val/0110.jpg",
            "data/icip/val/0110.jpg",
            "data/icip/val/0110.jpg",
            "data/icip/val/0110.jpg",
            "data/icip/val/0110.jpg",
            "data/icip/val/0110.jpg",
            "data/icip/val/0110.jpg",
        ]
    )
    print(predictions)
