# %%
import json
from pathlib import Path

from sqlitedict import SqliteDict

import wandb

api = wandb.Api()
runs = api.runs(
    path="hamgen/jetnet150_ddt",
)

# %%
fn = Path("~/fgsim/wandb.sqlite").expanduser()
with SqliteDict(fn, encode=json.dumps, decode=json.loads, autocommit=True) as db:
    for run in runs:
        db[runs[0].name] = runs[0].id

# %%
