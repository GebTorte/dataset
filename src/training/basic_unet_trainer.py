"""LWF-DLR U-Net segmentation training with SLURM submitit support."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.model import UNet
from src.dataset import S2TIFDataSet

from src.misc import select_patches_from_dataset

logger = logging.getLogger(__name__)




class LWFUNetTrainer:
    """Training class for LWF-DLR U-Net segmentation that can be submitted to SLURM.
    
    ADAPTED to s2 cloudsen satellitecloudgenerator dataset
    """

    def __init__(
        self,
        user: str,
        seed:int|None=42,
        repo: str = "cloudsen12", 
        csv_name: str = "cloudsen12_initial_cloudfree_dev.csv",
        num_classes: int = 3, # free, cloud, shadow
        in_channels: int = 12, # without cirrus
        base_channels: int = 32,
        epochs: int = 32,
        batch_size: int = 16,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        num_workers: int = 16,
        prefetch_factor: int = 8,
        val_split: float = 0.2,
        val_every_n_steps: int = 250,
        val_every_n_steps_warmup: int = 50,
        warmup_steps: int = 100,
        save_every_n_steps: int = 20,
        experiment_group: str = "LWF-DLR",
        experiment_id: str = "cloudsen12_baseline_unet_slurm",
    ):
        """
        Initialize the LWF U-Net trainer.

        Parameters
        ----------
        user : str
            Username for HPC paths.
        csv_name : str, optional
            Name of CSV file with info on e.g. cloudfree images
        num_classes : int, optional
            Number of output classes (default: 3).
        in_channels : int, optional
            Number of input channels (default: 1).
        base_channels : int, optional
            Base number of channels in U-Net (default: 32).
        epochs : int, optional
            Number of training epochs (default: 32).
        batch_size : int, optional
            Batch size for training (default: 12).
        lr : float, optional
            Initial learning rate (default: 0.001).
        weight_decay : float, optional
            Weight decay for optimizer (default: 0.0001).
        num_workers : int, optional
            Number of data loading workers (default: 16).
        prefetch_factor : int, optional
            Number of batches to prefetch (default: 8).
        val_split : float, optional
            Fraction of data for validation (default: 0.2).
        val_every_n_steps : int, optional
            Validate every N steps (default: 250).
        val_every_n_steps_warmup : int, optional
            Validate every N steps during warmup (default: 50).
        warmup_steps : int, optional
            Number of warmup steps (default: 100).
        save_every_n_steps : int, optional
            Save checkpoint every N steps (default: 20).
        experiment_group : str, optional
            Experiment group name (default: "LWF-DLR").
        experiment_id : str, optional
            Unique experiment identifier (default: "baseline_unet_slurm").
        """
        self.user = user
        self.seed=seed
        self.repo = repo
        self.csv_name = csv_name
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.val_split = val_split
        self.val_every_n_steps = val_every_n_steps
        self.val_every_n_steps_warmup = val_every_n_steps_warmup
        self.warmup_steps = warmup_steps
        self.save_every_n_steps = save_every_n_steps
        self.experiment_group = experiment_group
        self.experiment_id = experiment_id

        self.root_hpc = Path("/dss/dsstbyfs02/pn49ci/pn49ci-dss-0026")
        self.user_path = self.root_hpc / user
        self.data_path = self.user_path / repo / "data"
        self.data_root = self.data_path

        self.experiment_dir = self.user_path / f"experiments/{experiment_group}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir = self.experiment_dir / experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = None
        self.model = None
        self.writer = None
        self.train_loader = None
        self.val_loader = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

    def setup(self) -> None:
        """Set up device, model, and data loaders."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        # set random seed
        if self.seed:
            torch.manual_seed(self.seed)

        # Load dataset
        csv_path = self.data_path / self.csv_name
        file_names = select_patches_from_dataset(csv_path, self.data_root)

        # TODO: add shuffle to split?

        # Train/val split
        split_idx = int((1 - self.val_split) * len(file_names))
        train_names = file_names[:split_idx]
        val_names = file_names[split_idx:]

        logger.info("Total samples: %d", len(file_names))
        logger.info("Training samples: %d", len(train_names))
        logger.info("Validation samples: %d", len(val_names))

        # Create datasets and dataloaders
        # TODO: add seed etc to init params
        train_dataset = S2TIFDataSet(train_names, self.data_root, seed=self.seed)
        val_dataset = S2TIFDataSet(val_names, self.data_root, seed=self.seed)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size * 5,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
        )

        # Create model
        self.model = UNet(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_channels=self.base_channels,
        ).to(self.device)

        logger.info(
            "Model Parameters: %s",
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

        # Loss, optimizer, scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=0.0
        )

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.experiment_dir))

    def validate_model(self) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for inputs, targets in tqdm(
                self.val_loader, desc="Validation", leave=False
            ):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def save_checkpoint(
        self, epoch: int, global_step: int, val_loss: float, is_best: bool = False
    ) -> Path:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{global_step}.pth"

        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            checkpoint_path,
        )

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                best_path,
            )
            logger.info("Best model saved! (Val Loss: %.4f)", val_loss)

        return checkpoint_path

    def train(self) -> None:
        """Run the step-based training loop."""
        global_step = 0
        validation_cycle = 0
        best_val_loss = float("inf")

        # Training metrics accumulator (since last validation)
        train_running_loss_since_val = 0.0
        train_samples_since_val = 0

        logger.info("=" * 80)
        logger.info("TRAINING CONFIGURATION")
        logger.info("=" * 80)
        logger.info("Epochs: %d", self.epochs)
        logger.info("Step-based validation:")
        logger.info(
            "  - Warmup: every %d steps for first %d steps",
            self.val_every_n_steps_warmup,
            self.warmup_steps,
        )
        logger.info("  - Normal: every %d steps", self.val_every_n_steps)
        logger.info(
            "Checkpoint saving: every %d steps + best model", self.save_every_n_steps
        )
        logger.info("=" * 80)

        for epoch in range(self.epochs):
            logger.info("\n" + "=" * 60)
            logger.info("Epoch %d/%d", epoch + 1, self.epochs)
            logger.info("=" * 60)

            self.model.train()

            for batch_idx, (inputs, targets) in enumerate(
                tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
            ):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Accumulate training metrics
                batch_size = inputs.size(0)
                train_running_loss_since_val += loss.item() * batch_size
                train_samples_since_val += batch_size

                # Log learning rate every step
                self.writer.add_scalar(
                    "Train/learning_rate", self.optimizer.param_groups[0]["lr"], global_step
                )

                global_step += 1

                # STEP-BASED VALIDATION
                # Determine validation frequency (warmup vs normal)
                if global_step <= self.warmup_steps:
                    current_val_freq = self.val_every_n_steps_warmup
                else:
                    current_val_freq = self.val_every_n_steps

                # Check if it's time to validate
                if global_step % current_val_freq == 0:
                    validation_cycle += 1

                    # Log averaged training loss since last validation
                    if train_samples_since_val > 0:
                        train_loss_avg = (
                            train_running_loss_since_val / train_samples_since_val
                        )
                        self.writer.add_scalar(
                            "Train/total_loss_interval", train_loss_avg, global_step
                        )

                    # Reset training accumulators
                    train_running_loss_since_val = 0.0
                    train_samples_since_val = 0

                    # Run validation
                    val_loss = self.validate_model()

                    # TensorBoard Logging
                    self.writer.add_scalar("Val/total_loss", val_loss, global_step)

                    # Console output
                    logger.info(
                        "\n[Step %d | Cycle %d | Epoch %d]",
                        global_step,
                        validation_cycle,
                        epoch + 1,
                    )
                    logger.info("  Val Loss: %.4f", val_loss)

                    # Checkpoint Saving
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss

                    # Save checkpoint every N steps or if best
                    if global_step % self.save_every_n_steps == 0 or is_best:
                        self.save_checkpoint(
                            epoch, global_step, val_loss, is_best=is_best
                        )
                        if global_step % self.save_every_n_steps == 0:
                            logger.info("Checkpoint saved at step %d", global_step)

                    # Return to training mode
                    self.model.train()

            # Update learning rate at end of epoch
            self.scheduler.step()
            logger.info(
                "\nEpoch %d complete. LR: %.6f",
                epoch + 1,
                self.optimizer.param_groups[0]["lr"],
            )

        # Final Save
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info("Total steps: %d", global_step)
        logger.info("Validation cycles: %d", validation_cycle)
        logger.info("Best Val Loss: %.4f", best_val_loss)

        # Save final model
        final_path = self.checkpoint_dir / "final_model.pth"
        torch.save(
            {
                "epoch": self.epochs,
                "global_step": global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            },
            final_path,
        )
        logger.info("Final model saved to: %s", final_path)
        logger.info("Best model saved to: %s", self.checkpoint_dir / "best_model.pth")
        logger.info("TensorBoard logs: %s", self.experiment_dir)
        logger.info("To view logs, run: tensorboard --logdir=%s", self.experiment_dir)
        logger.info("=" * 60)

        # Close TensorBoard writer
        self.writer.close()

    def __call__(self) -> str:
        """
        Run the main training loop (called by submitit).

        Returns
        -------
        str
            Path to the saved final model.
        """
        logger.info("Starting training: %s", self.experiment_id)
        self.setup()
        self.train()
        return str(self.checkpoint_dir / "final_model.pth")