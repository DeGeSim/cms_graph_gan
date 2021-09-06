from typing import Dict, List, Union

import torch
import torch_geometric

BatchType = Union[torch_geometric.data.batch.Batch, Dict[str, torch.Tensor]]

DataSetType = List[BatchType]
