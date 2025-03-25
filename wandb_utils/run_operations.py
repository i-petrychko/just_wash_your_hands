import wandb
import os
from omegaconf import OmegaConf


def get_run_id(config_path):

    config = OmegaConf.load(config_path)
    project_name = config.wandb.project_name
    entity_name = config.wandb.entity
    run_name = config.wandb.name

    api = wandb.Api()

    runs = api.runs(
        path=f"{entity_name}/{project_name}", filters={"display_name": run_name}
    )

    sorted_runs = sorted(runs, key=lambda run: run.created_at, reverse=True)

    if sorted_runs:
        latest_run = sorted_runs[0]
        run_id = latest_run.id
        return run_id
    

def download_file(run_id, project_name, entity_name, file_name, output_path):
    api = wandb.Api()
    run = api.run(f"{entity_name}/{project_name}/{run_id}")
    run.file(file_name).download(root=output_path)


def upload_dir(run_id, project_name, entity_name, dir_path, base_path=None):
    api = wandb.Api()
    run = api.run(f"{entity_name}/{project_name}/{run_id}")
    run.save(dir_path, base_path=base_path)


def upload_file(run_id, project_name, entity_name, file_path):
    api = wandb.Api()
    run = api.run(f"{entity_name}/{project_name}/{run_id}")
    
    run.upload_file(file_path)

def upload_directory(run_id, project_name, entity_name, path_to_directory):

    if os.path.exists(path_to_directory) and os.path.isdir(path_to_directory):
        for root, dirs, files in os.walk(path_to_directory):
            for file in files:
                file_path = os.path.join(root, file)
                upload_file(run_id, project_name, entity_name, file_path)
    else:
        print(f"Directory '{path_to_directory}' does not exist.")

    

if __name__ == "__main__":
    val_path = "test_validation"
    upload_directory("2bnwcd14", "RT-DETR", "petrychko-vitalii-ukrainian-catholic-university", val_path)



