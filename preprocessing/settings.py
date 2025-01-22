from pathlib import Path
from typing import Tuple
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    pixel_size: float = Field(0.2506, env="PIXEL_SIZE")
    image_shape: Tuple[int, int, int] = Field((640, 640, 3), env="IMAGE_SHAPE")
    data_path: Path = Field(Path("./data"), env="DATASET_PATH")

    @validator("image_shape", pre=True)
    def parse_image_shape(cls, value):
        if isinstance(value, str):
            return tuple(map(int, value.strip("()").split(",")))
        return value


settings = Settings()

