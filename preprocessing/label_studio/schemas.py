from pydantic import BaseModel
from typing import List, Optional, Any, Union
import sys
from enum import Enum


sys.path.append(".")

from common.utils import read_json

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
    model: Optional[str] = None
    status: Status = Status.PENDING


class Value(BaseModel):
    x: float
    y: float
    width: float
    height: float
    rotation: Optional[int] = 0
    rectanglelabels: List[str]


class Result(BaseModel):
    original_width: Optional[int] = None
    original_height: Optional[int] = None
    image_rotation: Optional[int] = None
    value: Value
    id: Optional[str] = None
    from_name: str
    to_name: str
    type: str = "rectanglelabels"
    origin: Optional[str] = None


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


class Choice(str, Enum):
    REJECT = "Reject"
    APPROVE = "Approve"


class Choices(BaseModel):
    choices: List[Choice]


class ResultOutput1(BaseModel):
    value: Choices
    id: str
    from_name: str
    to_name: str
    type: str
    origin: str


class PredictionOutput(BaseModel):
    id: int
    result: List[Result]
    model_version: str
    created_ago: str
    score: Optional[int] = None
    cluster: Optional[int] = None
    neighbors: Optional[int] = None
    mislabeling: float
    created_at: str
    updated_at: str
    model: Optional[str] = None
    model_run: Optional[int] = None
    task: int
    project: int


class Annotation(BaseModel):
    id: int
    completed_by: int
    result: List[Union[Result, ResultOutput1]] = []
    was_cancelled: bool
    ground_truth: bool
    created_at: str
    updated_at: str
    draft_created_at: Optional[str] = None
    lead_time: float
    prediction: PredictionOutput
    result_count: int
    unique_id: str
    import_id: Optional[int]
    last_action: Optional[str]
    task: int
    project: int
    updated_by: int
    parent_prediction: int
    parent_annotation: Optional[int]
    last_created_by: Optional[int]


class FilteredOutput(BaseModel):
    id: int
    annotations: List[Annotation]
    file_upload: str
    drafts: List
    predictions: List[int]
    data: Data
    meta: dict
    created_at: str
    updated_at: str
    inner_id: int
    total_annotations: int
    cancelled_annotations: int
    total_predictions: int
    comment_count: int
    unresolved_comment_count: int
    last_comment_updated_at: Optional[Any] = None
    project: int
    updated_by: int
    comment_authors: List

    @classmethod
    def from_dict(cls, filtered_output: dict):

        return cls(**filtered_output)


class FilteredLabel(BaseModel):
    id: int
    img_path: str
    results: List[Result]
    predictions: List[Result]
    choice: Choice

    @classmethod
    def from_filtered_output(cls, filtered_output: FilteredOutput):

        id = filtered_output.id
        img_path = filtered_output.data.img_path
        results = [
            result
            for annotation in filtered_output.annotations
            for result in annotation.result
            if isinstance(result, Result)
        ]
        predictions = [
            prediction_result
            for annotation in filtered_output.annotations
            for prediction_result in annotation.prediction.result
        ]
        choice = [
            result.value.choices[0]
            for annotation in filtered_output.annotations
            for result in annotation.result
            if isinstance(result, ResultOutput1)
        ][0]
        return cls(
            id=id,
            img_path=img_path,
            results=results,
            predictions=predictions,
            choice=choice,
        )


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

    filtered_json_output = read_json(
        "preprocessing/label_studio/results/Paragonimus_spp_all.json"
    )
    filtered_output = FilteredOutput.from_dict(filtered_json_output[0])
    filtered_label = FilteredLabel.from_filtered_output(filtered_output)
    print(filtered_label)
