from typing import List, Optional, Union, Dict
from pydantic import BaseModel, Field


# Subschema for train, validation, and test sets
class DatasetSubsetConfig(BaseModel):
    use_empty_txt: Optional[bool] = Field(
        False, description="Whether to include empty text files in this subset"
    )
    min_relative_area: Optional[float] = Field(
        0.0, description="Minimum relative area for filtering in this subset"
    )
    max_relative_area: Optional[float] = Field(
        1.0, description="Maximum relative area for filtering in this subset"
    )
    min_scaling_cf: Optional[Union[float, str]] = Field(
        "-inf", description="Minimum scaling coefficient"
    )
    max_scaling_cf: Optional[Union[float, str]] = Field(
        "+inf", description="Maximum scaling coefficient"
    )


# Subschema for filtering categories
class CategoryConfig(BaseModel):
    name: Optional[str] = Field(None, description="Name of the category to filter")
    train_set: Optional[DatasetSubsetConfig] = Field(
        None, description="Configuration for the training subset"
    )
    val_set: Optional[DatasetSubsetConfig] = Field(
        None, description="Configuration for the validation subset"
    )
    test_set: Optional[DatasetSubsetConfig] = Field(
        None, description="Configuration for the testing subset"
    )


# Schema for filtering configuration
class FilteringConfig(BaseModel):
    label_statuses: Optional[List[str]] = Field(
        default=[], description="List of label statuses to filter"
    )
    categories: Optional[List[CategoryConfig]] = Field(
        default=[], description="List of category configurations"
    )

    def get_set_config_dict(self, set_type: str) -> Dict[str, DatasetSubsetConfig]:
        set_config_dict = {}

        for category in self.categories:
            if category.name:
                # Using getattr to dynamically get the set type (train_set, val_set, test_set)
                subset_config = getattr(category, set_type, None)

                if subset_config:
                    set_config_dict[category.name] = subset_config

        return set_config_dict


# Schema for preprocessing configuration
class PreprocessingConfig(BaseModel):
    out_channels: Optional[int] = Field(
        None, description="Number of output channels for preprocessing"
    )
    out_dim: Optional[List[int]] = Field(
        None, description="Output dimensions of the image (width, height)"
    )
    pixel_size: Optional[float] = Field(
        None, description="Pixel size for scaling images to actual sizes in microns"
    )


# Schema for paths configuration
class PathsConfig(BaseModel):
    dataset_path: Optional[str] = Field(None, description="Path to the dataset")
    out_path: Optional[str] = Field(None, description="Output path template")


# Schema for split configuration
class SplitConfig(BaseModel):
    seed: Optional[int] = Field(None, description="Random seed for data splitting")
    type: Optional[str] = Field(None, description="Type of split (e.g., stratified)")
    ratio: Optional[List[float]] = Field(
        None, description="Proportions for train, validation, and test sets"
    )


# Main schema for the entire configuration
class Config(BaseModel):
    version: Optional[Union[str, float]] = Field(
        None, description="Version of the configuration"
    )
    preprocessing: Optional[PreprocessingConfig] = Field(
        None, description="Preprocessing settings"
    )
    paths: Optional[PathsConfig] = Field(None, description="Paths configuration")
    split: Optional[SplitConfig] = Field(None, description="Split configuration")
    filtering: Optional[FilteringConfig] = Field(
        None, description="Filtering configuration"
    )
