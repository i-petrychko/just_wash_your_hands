from pathlib import Path
from typing import Tuple
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    pixel_size: float = Field(0.2506, env="PIXEL_SIZE")
    image_shape: Tuple[int, int, int] = Field((640, 640, 3), env="IMAGE_SHAPE")
    data_path: Path = Field(Path("./data/dataset"), env="DATASET_PATH")
    preprocessing_config_path: Path = Field(
        Path("./preprocessing/config.yaml"), env="PREPROCESSING_CONFIG_PATH"
    )

    orig_icip_train_images_dir_path: Path = Field(
        Path("./data/icip/Chula-ParasiteEgg-11/data"), env="ICIP_TRAIN_IMAGES_DIR_PATH"
    )
    orig_icip_train_labels_file_path: Path = Field(
        Path("./data/icip/Chula-ParasiteEgg-11/labels.json"),
        env="ICIP_TRAIN_LABELS_FILE_PATH",
    )
    orig_icip_test_images_dir_path: Path = Field(
        Path("./data/icip/test/data"), env="ICIP_TEST_IMAGES_DIR_PATH"
    )
    orig_icip_test_labels_file_path: Path = Field(
        Path("./data/icip/test/test_labels_200.json"), env="ICIP_TEST_LABELS_FILE_PATH"
    )

    @validator("image_shape", pre=True)
    def parse_image_shape(cls, value):
        if isinstance(value, str):
            return tuple(map(int, value.strip("()").split(",")))
        return value


settings = Settings()
