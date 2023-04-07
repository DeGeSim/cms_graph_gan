import os
from pathlib import Path
from shutil import copytree

from hyperpars import hyperpars, project_name

import wandb
from fgsim.commands.setup import filter_paths

os.chdir(Path("~").expanduser())
sweep_path = Path(f"~/fgsim/wd/{project_name}").expanduser()
os.makedirs(sweep_path, exist_ok=True)
# Define sweep config
sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "w1m"},
    "parameters": hyperpars,
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
with open(sweep_path / "sweepid", "w") as f:
    f.write(sweep_id)

copytree(
    "fgsim",
    sweep_path / "fgsim",
    ignore=lambda d, files: [f for f in files if filter_paths(d, f)],
    dirs_exist_ok=True,
)
