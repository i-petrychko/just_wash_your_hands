from preprocessing.schemas import PreprocessingSchema
from preprocessing.settings import settings


def get_objects_characteristics():
    return PreprocessingSchema.from_yaml(settings.preprocessing_config_path)
