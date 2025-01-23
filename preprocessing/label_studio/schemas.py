from pydantic import BaseModel
from typing import List
import sys


sys.path.append(".")

from preprocessing.schemas import (
    ImageLabelSchema,
    LabelSchema,
    ObjectSchema,
    LabelCoordinatesSchema,
    ImageShape,
    Status,
)


class Data(BaseModel):
    image: str  # label studio format path
    img_path: str
    iou: float = 0.0
    gt_iou: float = 0.0
    confidence: float = 0.0
    model: str = None
    status: Status = Status.PENDING


class Value(BaseModel):
    x: float
    y: float
    width: float
    height: float
    rectanglelabels: List[str]


class Result(BaseModel):
    from_name: str
    to_name: str
    type: str = "rectanglelabels"
    value: Value


class Prediction(BaseModel):
    model_version: str
    result: List[Result]


class Label(BaseModel):
    data: Data
    predictions: List[Prediction]

    @classmethod
    def from_image_label(cls, image_label: ImageLabelSchema):
        """
        Convert an ImageLabelSchema instance into a Label instance.
        """
        # Prepare the data object
        data = Data(
            image=f"/data/local-files/?d={image_label.img_path}",
            img_path=image_label.img_path,
            status=image_label.status,
            confidence=(
                min(
                    [
                        label.confidence
                        for label in image_label.labels
                        if label.confidence is not None
                    ]
                )
                if image_label.labels
                and any(label.confidence is not None for label in image_label.labels)
                else 0.0
            ),
        )

        # Prepare the predictions based on the labels in ImageLabelSchema
        predictions = []
        for label_schema in image_label.labels:
            yolo = label_schema.yolo_annotation
            width = yolo.width * 100  # Scale width by 100
            height = yolo.height * 100  # Scale height by 100
            x = (yolo.mid_x - yolo.width / 2) * 100  # Convert mid_x to upper-left x
            y = (yolo.mid_y - yolo.height / 2) * 100  # Convert mid_y to upper-left y
            result = Result(
                from_name="gt_labels",
                to_name="gt_image",
                value=Value(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    rectanglelabels=[label_schema.object.name],
                ),
            )
            predictions.append(
                Prediction(
                    model_version="ground_truth",
                    result=[result],
                )
            )

        return cls(data=data, predictions=predictions)


if __name__ == "__main__":
    example_image_schema = ImageLabelSchema(
        img_path=f"data/dataset/images/0.jpg",
        labels=[
            LabelSchema(
                object=ObjectSchema.from_initial_name("Taenia spp. egg"),
                coordinates=LabelCoordinatesSchema.from_xywh(100, 100, 500, 500),
                image_shape=ImageShape(width=1000, height=1000),
            )
        ],
    )

    print(Label.from_image_label(example_image_schema))
