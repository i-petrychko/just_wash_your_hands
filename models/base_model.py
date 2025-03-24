import torch
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class ImagePrediction:
    labels: List[int]
    normalized_cxcywhs: List[Tuple]
    confidences: List[float]
    img_path: str

    def model_dump(self) -> dict:
        return {
            "labels": self.labels,
            "normalized_cxcywhs": self.normalized_cxcywhs,
            "confidences": self.confidences,
            "img_path": self.img_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImagePrediction":
        return cls(
            labels=data["labels"],
            normalized_cxcywhs=data["normalized_cxcywhs"],
            confidences=data["confidences"],
            img_path=data["img_path"],
        )


class BaseModel:
    def __init__(self) -> None:
        self.device = self.get_device()

    def predict(self, paths, iou, min_conf, batch_size) -> List[ImagePrediction]:
        raise NotImplementedError

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
