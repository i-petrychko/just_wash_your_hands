import yaml
from pydantic import BaseModel, model_validator
from typing import List, Tuple, Optional, Union, Any
from uuid import uuid4
import sys
import json
from enum import Enum

sys.path.append(".")

from preprocessing.settings import settings
from common.utils import save_unserializable_json, read_json


class Choice(str, Enum):
    REJECT = "Reject"
    APPROVE = "Approve"


class Status(str, Enum):
    PENDING = "Pending"
    APPROVED = "Approved"
    APPROVED_AUTOMATICALLY = "Approved automatically"
    REJECTED = "Rejected"
    REJECTED_AUTOMATICALLY = "Rejected automatically"


def choice_to_status(choice: Choice) -> Status:
    if choice == Choice.APPROVE:
        return Status.APPROVED
    elif choice == Choice.REJECT:
        return Status.REJECTED
    else:
        raise ValueError(f"Invalid choice: {choice}")


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


class CharacteristicValueSchema(BaseModel):
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class YoloCoordinatesSchema(BaseModel):
    mid_x: float
    mid_y: float
    width: float
    height: float


class ImageShape(BaseModel):
    width: int
    height: int
    channels: int = 3


class ObjectSchema(BaseModel):
    name: str
    initial_name: Optional[str] = None
    bbox_color: Tuple[int, int, int] = (255, 255, 255)
    width: CharacteristicValueSchema = CharacteristicValueSchema()  # in microns
    height: CharacteristicValueSchema = CharacteristicValueSchema()  # in microns
    category_id: int

    @classmethod
    def from_initial_name(
        cls, initial_name: str, config_path: str = settings.preprocessing_config_path
    ):
        """
        Create an ObjectSchema instance using initial_name and load the corresponding data from the config.
        """
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        # Find the object in the config by initial_name
        for obj in config["objects"]:
            if obj["initial_name"] == initial_name:
                # Return an ObjectSchema instance using the found config data
                return cls(**obj)
        raise ValueError(
            f"Object with initial_name '{initial_name}' not found in config."
        )

    @classmethod
    def from_name(
        cls, name: str, config_path: str = settings.preprocessing_config_path
    ):
        """
        Create an ObjectSchema instance using initial_name and load the corresponding data from the config.
        """
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        # Find the object in the config by initial_name
        for obj in config["objects"]:
            if obj["name"] == name:
                # Return an ObjectSchema instance using the found config data
                return cls(**obj)
        raise ValueError(f"Object with name '{name}' not found in config.")


class LabelCoordinatesSchema(BaseModel):
    min_x: int
    max_x: int
    min_y: int
    max_y: int

    @classmethod
    def from_xywh(cls, x: float, y: float, width: float, height: float):
        return cls(
            min_x=int(x), max_x=int(x + width), min_y=int(y), max_y=int(y + height)
        )


class Data(BaseModel):
    image: str  # label studio format path
    img_path: str
    iou: float = 0.0
    gt_iou: float = 0.0
    confidence: float = 0.0
    model: Optional[str] = None
    status: Status = Status.PENDING


class LabelSchema(BaseModel):
    object: ObjectSchema
    uuid: str = str(uuid4())
    image_shape: ImageShape
    confidence: Optional[float] = None
    pixel_size: Optional[float] = None
    coordinates: LabelCoordinatesSchema = None
    yolo_annotation: YoloCoordinatesSchema = None
    num_vertices: Optional[int] = None
    relative_area: Optional[float] = None
    scaling_coef: Optional[float] = None

    @classmethod
    def from_result(cls, result: Result):
        object = ObjectSchema.from_name(result.value.rectanglelabels[0])
        image_shape = ImageShape(
            width=result.original_width, height=result.original_height
        )
        coordinates = LabelCoordinatesSchema.from_xywh(
            x=result.value.x * result.original_width / 100,
            y=result.value.y * result.original_height / 100,
            width=result.value.width * result.original_width / 100,
            height=result.value.height * result.original_height / 100,
        )
        return cls(object=object, image_shape=image_shape, coordinates=coordinates)

    @model_validator(mode="after")
    def calculate_yolo_and_area(cls, values):
        coordinates = values.coordinates
        image_shape = values.image_shape

        if coordinates and image_shape:
            # Calculate YOLO annotation
            mid_x = ((coordinates.min_x + coordinates.max_x) / 2) / image_shape.width
            mid_y = ((coordinates.min_y + coordinates.max_y) / 2) / image_shape.height
            width = (coordinates.max_x - coordinates.min_x) / image_shape.width
            height = (coordinates.max_y - coordinates.min_y) / image_shape.height

            # Set YOLO annotation
            values.yolo_annotation = YoloCoordinatesSchema(
                mid_x=mid_x, mid_y=mid_y, width=width, height=height
            )
            # Set relative area
            values.relative_area = width * height

        return values


class Choices(BaseModel):
    choices: List[Choice]


class Prediction(BaseModel):
    model_version: str
    result: List[Result]


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


class ImageLabelSchema(BaseModel):
    img_path: str
    status: Status = Status.PENDING
    labels: List[LabelSchema]

    def save_to_json(self, path: str):
        with open(path, "w") as file:
            json.dump(self.model_dump(), file, indent=4)

    @classmethod
    def from_dict(cls, labels: dict):
        return cls.model_validate(labels)

    @classmethod
    def from_filtering_results(cls, filtered_label: FilteredLabel):

        status = choice_to_status(filtered_label.choice)
        img_path = filtered_label.img_path
        labels = []
        for result in filtered_label.results:
            label = LabelSchema.from_result(result)
            labels.append(label)

        return cls(status=status, img_path=img_path, labels=labels)


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


class PreprocessingSchema(BaseModel):
    objects: List[ObjectSchema]

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """
        Load YAML content and initialize the schema.
        """
        with open(yaml_path, "r") as file:
            yaml_content = yaml.safe_load(file)
        return cls(**yaml_content)


if __name__ == "__main__":
    preprocessing_config = PreprocessingSchema.from_yaml("preprocessing/config.yaml")
    print(preprocessing_config)

    example_image_schema = [
        ImageLabelSchema(
            img_path=f"path_to_image.png",
            labels=[
                LabelSchema(
                    object=ObjectSchema.from_initial_name("Taenia spp. egg"),
                    coordinates=LabelCoordinatesSchema.from_xywh(100, 100, 500, 500),
                    image_shape=ImageShape(width=1000, height=1000),
                )
            ],
        )
    ]

    save_unserializable_json(example_image_schema, "output.json")

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
