import yaml
from pydantic import BaseModel, EmailStr, ValidationError
from typing import List
from uuid import uuid4

class CharacteristicValueSchema(BaseModel):
    min_value: float = None
    max_value: float = None

class ObjectSchema(BaseModel):
    name: str
    initial_name: str
    width: CharacteristicValueSchema = CharacteristicValueSchema() # in microns
    height: CharacteristicValueSchema = CharacteristicValueSchema() # in microns
    id: int

class LabelCoordinatesSchema(BaseModel):
    min_x: int
    max_x: int
    min_y: int
    max_y: int

class YoloCoordinatesSchema(BaseModel):
    mid_x: float
    mid_y: float
    width: float
    height: float

class LabelSchema(BaseModel):
    object: ObjectSchema
    uuid: str = uuid4()
    pixel_size: float
    coordinates: LabelCoordinatesSchema = None
    num_vertices: int
    relative_area: float
    scaling_coef: float
    yolo_annotation: YoloCoordinatesSchema = None
    filtered_manually: bool = False

class ImageLabelSchema(BaseModel):
    img_path: str
    labels: List[LabelSchema]


class PreprocessingSchema(BaseModel):
    objects: List[ObjectSchema]

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """
        Load YAML content and initialize the schema.
        """
        with open(yaml_path, 'r') as file:
            yaml_content = yaml.safe_load(file)
        return cls(**yaml_content)

if __name__=="__main__":
    preprocessing_schema = PreprocessingSchema.from_yaml("preprocessing/config.yaml")
    print(preprocessing_schema)


