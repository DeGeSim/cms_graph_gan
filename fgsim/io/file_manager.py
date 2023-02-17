from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

from fgsim.config import conf


# Make sure the readpath takes a path and start and end of the chunk
# It loads a list of files, and then loads the lengths of those files
class FileManager:
    def __init__(
        self, path_to_len: Callable[[Path], int], files: Optional[List[Path]] = None
    ) -> None:
        self._path_to_len = path_to_len
        self.files = files
        if self.files is None:
            self.files: List[Path] = self._get_file_list()
        self.file_len_dict: Dict[Path, int] = self._load_len_dict()

    def _get_file_list(self) -> List[Path]:
        ds_path = Path(conf.path.dataset).expanduser()
        assert ds_path.is_dir()
        files = sorted(ds_path.glob(conf.loader.dataset_glob))
        if len(files) < 1:
            raise RuntimeError("No datasets found")
        return [f for f in files]

    def _load_len_dict(self) -> Dict[Path, int]:
        if not Path(conf.path.ds_lenghts).is_file():
            self.save_len_dict()
        with open(conf.path.ds_lenghts, "r") as f:
            len_dict: Dict[Path, int] = {
                Path(k): int(v)
                for k, v in yaml.load(f, Loader=yaml.SafeLoader).items()
            }
        return len_dict

    def save_len_dict(self) -> None:
        self.len_dict = {}
        for fn in self.files:
            self.len_dict[str(fn)] = self._path_to_len(fn)
        ds_processed = Path(conf.path.dataset_processed)
        if not ds_processed.is_dir():
            ds_processed.mkdir()
        with open(conf.path.ds_lenghts, "w") as f:
            yaml.dump(self.len_dict, f, Dumper=yaml.SafeDumper)
