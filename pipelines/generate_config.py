import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import yaml

from src.core import YAMLConfig 


def main(args, ) -> None:

    config_path = f"{args.config_path}/{args.config}.yml"

    cfg = YAMLConfig(
        config_path
    )

    with open("config.yaml", "w") as f:
        yaml.dump(cfg.yaml_cfg, f, default_flow_style=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--config_path', '-p', type=str, default="recipes/")
    args = parser.parse_args()

    main(args)
