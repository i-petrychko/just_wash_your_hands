from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf
import argparse
import os
import sys

sys.path.append(".")

from schemas.preprocessing import Config
from common.utils import save_unserializable_json, read_json
from pipelines.utils import convert_coco_to_image_labels



def main(args):

    config = OmegaConf.load(args.config_path)

    config = OmegaConf.to_container(config, resolve=True)
    config = Config(**config)

    output_path = f"{config.paths.out_path}/splits"
    os.makedirs(output_path, exist_ok=True)

    # train set split into train and val
    train_labels_json = read_json(f"{config.paths.dataset_path}/annotations/train.json")
    initial_train_data = convert_coco_to_image_labels(train_labels_json, f"{config.paths.dataset_path}/train")
    initial_train_labels = [label.labels[0].object.name for label in initial_train_data]

    if config.train_split.type == "stratified":
        train_data, val_data, _, _ = train_test_split(
            initial_train_data,
            initial_train_labels,
            test_size=config.train_split.ratio[1],
            random_state=config.seed,
        )
    else:
        print("Unknown split type")

    save_unserializable_json(train_data, f"{output_path}/train.json")
    save_unserializable_json(val_data, f"{output_path}/val.json")

    # save test set
    test_labels_json = read_json(f"{config.paths.dataset_path}/annotations/test.json")
    test_data = convert_coco_to_image_labels(test_labels_json, f"{config.paths.dataset_path}/test")
    save_unserializable_json(test_data, f"{output_path}/test.json")


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, default="config.yaml")
    args = parser.parse_args()

    main(args)
