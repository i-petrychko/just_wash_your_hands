import json


def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_json(data, path):
    with open(path, "w") as file:
        json.dump(data, file, indent=4)


def save_unserializable_json(data, path):
    """
    Save a list of Pydantic models to a JSON file.
    """
    with open(path, "w") as file:
        json.dump([item.model_dump() for item in data], file, indent=4)
