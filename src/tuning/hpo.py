import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


from src.dataset import S2TIFDataSet, S2TIFDataSet512
from src.misc import select_patches_from_dataset


logger = logging.getLogger(__name__)



def objective(trial, train_loader):
    hidden_size = trial.suggest_int('hidden_size', 128, 512)

    learning_rate = trial.suggest_float('lr', 1e-4, 1e-1, log=True)

    bnm = trial.suggest_float('momentum', 0.1, 0.9)


    transform = transforms.Compose([transforms.ToTensor()])

    # Load dataset
    csv_path = self.data_path / self.csv_name
    file_names = select_patches_from_dataset(csv_path, self.data_root)

    # Train/val split
    split_idx = int((1 - self.val_split) * len(file_names))
    train_names = file_names[:split_idx]
    val_names = file_names[split_idx:]

    logger.info("Total samples: %d", len(file_names))
    logger.info("Training samples: %d", len(train_names))
    #logger.info("Validation samples: %d", len(val_names))

    # Create datasets and dataloaders
    train_dataset = S2TIFDataSet(train_names, self.data_root, seed=self.seed)
    #val_dataset = S2TIFDataSet(val_names, self.data_root, seed=self.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=12,
        shuffle=True,
        num_workers=16,
        prefetch_factor=8,
        drop_last=True,
        pin_memory=True,
    )

    model = UNet(hidden_size)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    return loss.item()


if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)
    print("Best Hyperparameters:", study.best_params)