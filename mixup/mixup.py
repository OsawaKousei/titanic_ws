from typing import Tuple

import pandas as pd
import torch
from torch import utils
from torch.utils import data


class MyDataset(data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str = "y",
    ) -> None:
        super().__init__()
        self.features = df.drop(columns=[target_column]).values
        self.targets = df[[target_column]].values
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.tensor(self.features[idx, ...], dtype=torch.float)
        features = features.to(self.device, non_blocking=True)

        target = torch.tensor(self.targets[idx, ...], dtype=torch.float)
        target = target.to(self.device, non_blocking=True)

        return features, target
