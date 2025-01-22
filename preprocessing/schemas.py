import yaml
from pydantic import BaseModel, model_validator, field_validator
from typing import List, Tuple
from uuid import uuid4
import sys
import json

sys.path.append(".")

from preprocessing.settings import settings
from common.utils import save_unserializable_json


class CharacteristicValueSchema(BaseModel):
    min_value: float = 0
    max_value: float = 0


class ObjectSchema(BaseModel):
    name: str
    initial_name: str
    bbox_color: Tuple[int, int, int] = (255, 255, 255)
    width: CharacteristicValueSchema = CharacteristicValueSchema()  # in microns
    height: CharacteristicValueSchema = CharacteristicValueSchema()  # in microns
    category_id: int

    @field_validator("name", mode="before")
    def set_initial_name(cls, value, values):
        """
        Set initial_name to name if initial_name is not provided.
        """
        if value is None:  # if initial_name is not provided
            return values.get("initial_name", "")  # fallback to 'name'
        return value

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


class YoloCoordinatesSchema(BaseModel):
    mid_x: float
    mid_y: float
    width: float
    height: float


class ImageShape(BaseModel):
    width: int
    height: int
    channels: int = 3


class LabelSchema(BaseModel):
    object: ObjectSchema
    uuid: str = str(uuid4())
    image_shape: ImageShape
    confidence: float = None
    pixel_size: float = None
    coordinates: LabelCoordinatesSchema = None
    yolo_annotation: YoloCoordinatesSchema = None
    num_vertices: int = None
    relative_area: float = None
    scaling_coef: float = None
    filtered_manually: bool = False

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


class ImageLabelSchema(BaseModel):
    img_path: str
    labels: List[LabelSchema]

    def save_to_json(self, path: str):
        with open(path, "w") as file:
            json.dump(self.model_dump(), file, indent=4)

    @classmethod
    def read_from_json(cls, path: str):
        with open(path, "r") as file:
            data = json.load(file)
        return cls.model_validate(data)  # Validate and reconstruct the object from JSON


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
