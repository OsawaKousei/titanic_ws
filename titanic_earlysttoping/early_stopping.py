from typing import Optional

import numpy as np
import torch


class EarlyStopping:

    def __init__(self, patience: int = 7, verbose: bool = False) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.force_cancel = False

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module) -> None:
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(
            model.state_dict(), "./titanic_earlysttoping/models/model.pth"
        )
        self.val_loss_min = val_loss
