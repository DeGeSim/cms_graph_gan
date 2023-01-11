import json
from pathlib import Path

from sqlitedict import SqliteDict


class ExperimentOrganizer:
    def __init__(self, name) -> None:
        self.fn = Path(f"~/fgsim/{name}.sqlite").expanduser()
        # with open(self.fn, "r") as f:
        #     self.d = yaml.load(f, Loader=yaml.SafeLoader)

    def __getitem__(self, key: str) -> str:
        with SqliteDict(
            self.fn, encode=json.dumps, decode=json.loads, autocommit=True
        ) as db:
            value = db[key]
        return value

    def __setitem__(self, key: str, value: str):
        with SqliteDict(
            self.fn, encode=json.dumps, decode=json.loads, autocommit=True
        ) as db:
            db[key] = value

    def __delitem__(self, key):
        with SqliteDict(
            self.fn, encode=json.dumps, decode=json.loads, autocommit=True
        ) as db:
            del db[key]

    def keys(self):
        with SqliteDict(
            self.fn, encode=json.dumps, decode=json.loads, autocommit=True
        ) as db:
            keys = list(db.keys())
        return keys

    # def recreate(self):
    #     workspace = comet_conf.workspace
    #     experiments = []
    #     for project in comet_api.get_projects(workspace):
    #         experiments = experiments + comet_api.get(workspace, project)
    #     with SqliteDict(
    #         self.fn, encode=json.dumps, decode=json.loads, autocommit=True
    #     ) as db:
    #         for key in db.keys():
    #             del db[key]
    #         for exp in experiments:
    #             db[exp.name] = exp.id
