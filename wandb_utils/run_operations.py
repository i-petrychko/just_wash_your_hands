import wandb
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

    

if __name__ == "__main__":
    run_id = get_run_id("recipes/rtdetr_r18vd_6x_icip.yml")
    print(run_id)

    download_file(run_id, "RT-DETR", "petrychko-vitalii-ukrainian-catholic-university", "run_output/checkpoint_latest.pth", "test_download")



