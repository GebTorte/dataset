from src.training.lwf_unet_aspp_trainer import *

from src.dataset import TestS2TIFDataSet, S2TIFDataSet

from torch.utils.data import DataLoader, RandomSampler

import optuna

logger = logging.getLogger(__name__)



class LWFUNetASPPOptunaCS12Val:
    """Optuna hpo class for LWF-DLR U-Net with ASPP + Residual + Weighted CE + Dice Loss."""

    def __init__(
        self,
        user: str,
        repo: str = "cloudsen12", 
        train_csv_name: str = "cloudsen12_initial_cloudfree_scribble.csv",
        val_csv_name: str = "cloudsen12_initial_high_test.csv",
        seed:int | None = 42,
        num_classes: int = 4,
        in_channels: int = 12,
        base_channels: int = 32,
        epochs: int = 3,
        batch_size: int = 12,
        n_trials:int = 5,
        n_gpus:int=1, 
        num_workers: int = 16,
        prefetch_factor: int = 8,
        val_split: float = 0.2,
        val_every_n_steps: int = 250,
        val_every_n_steps_warmup: int = 50,
        warmup_steps: int = 100,
        save_every_n_steps: int = 20,
        experiment_group: str = "LWF-DLR",
        experiment_id: str = "cloudsen12_aspp_residual_slurm",
        use_class_weights: bool = True,
        class_weights: list[float] | None = None,
        use_dice_loss: bool = True,
        dice_loss_weight: float = 0.3,
        dice_loss_smooth: float = 1.0,
        dice_loss_classes: list[int] | None = None,
        use_residual: bool = True,
        use_aspp: bool = True,
        aspp_rates: tuple[int, ...] = (3, 6, 9, 12),
    ):
        """
        Initialize the LWF U-Net trainer with ASPP and advanced loss functions.

        Parameters
        ----------
        user : str
            Username for HPC paths.
        csv_name : str, optional
            Name of CSV file with NPZ paths (default: "full_set_file.csv").
        num_classes : int, optional
            Number of output classes (default: 4).
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
            Unique experiment identifier (default: "aspp_residual_slurm").
        use_class_weights : bool, optional
            Whether to use class weights for CE loss (default: True).
        class_weights : list[float] | None, optional
            Class weights for CE loss (default: [1.0, 50.0, 5.0]).
        use_dice_loss : bool, optional
            Whether to use Dice loss (default: True).
        dice_loss_weight : float, optional
            Weight for Dice loss term (default: 0.3).
        dice_loss_smooth : float, optional
            Smoothing factor for Dice loss (default: 1.0).
        dice_loss_classes : list[int] | None, optional
            Classes to apply Dice loss to (None = all classes).
        use_residual : bool, optional
            Whether to use residual connections (default: True).
        use_aspp : bool, optional
            Whether to use ASPP in bottleneck (default: True).
        aspp_rates : tuple[int, ...], optional
            Dilation rates for ASPP (default: (3, 6, 9, 12)).
        """
        self.user = user
        self.repo = repo
        self.seed=seed
        self.train_csv_name = train_csv_name
        self.val_csv_name = val_csv_name
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.epochs = epochs
        self.batch_size = batch_size
        #self.lr = lr
        #self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.val_split = val_split
        self.val_every_n_steps = val_every_n_steps
        self.val_every_n_steps_warmup = val_every_n_steps_warmup
        self.warmup_steps = warmup_steps
        self.save_every_n_steps = save_every_n_steps
        self.experiment_group = experiment_group
        self.experiment_id = experiment_id

        self.n_trials = n_trials
        self.n_gpus = n_gpus

        # Loss configuration
        self.use_class_weights = use_class_weights
        self.class_weights = class_weights or [1.] * num_classes
        self.use_dice_loss = use_dice_loss
        self.dice_loss_weight = dice_loss_weight
        self.dice_loss_smooth = dice_loss_smooth
        self.dice_loss_classes = dice_loss_classes

        # Architecture configuration
        self.use_residual = use_residual
        self.use_aspp = use_aspp
        self.aspp_rates = aspp_rates

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
        self.criterion_ce = None
        self.criterion_dice = None
        self.optimizer = None
        self.scheduler = None

    def objective(self, trial) -> None:
        self.setup(trial)

        return self.train()


    def setup(self, trial) -> None:
        """Set up device, model, and data loaders."""
        logger.info("Using device: %s", self.device)

        # set random seed
        if self.seed:
            torch.manual_seed(self.seed)

        # Load datasets
        csv_path = self.data_path / self.train_csv_name
        file_names = select_patches_from_dataset(csv_path, self.data_root, type_folder="")
        val_file_names = select_patches_from_dataset(
            self.data_path / self.val_csv_name, self.data_root, type_folder=""
        )

        # Train/val split (atm training on full set and validating on high gt)
        #split_idx = int((1 - self.val_split) * len(file_names))
        train_names = file_names # file_names[:split_idx]
        val_names = val_file_names # file_names[split_idx:]

        #logger.info("Total samples: %d", len(file_names))
        logger.info("Training samples: %d", len(train_names))
        logger.info("Validation samples: %d", len(val_names))

        # Create datasets and dataloaders
        # TODO: add data params to trial/study
        train_dataset = S2TIFDataSet(train_names, self.data_root, seed=self.seed)
        val_dataset = TestS2TIFDataSet(val_names, seed=self.seed)
        #val_dataset = S2TIFDataSet(val_names, self.data_root, seed=self.seed)

        # optuna suggest hparams:
        lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
        #batch_size = 12 # trial.suggest_categorical('batch_size', [8, 12, 16])
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
        bn_momentum = trial.suggest_categorical('batch_norm_momentum', [0.1, 0.9, 0.99])

        # deprecate
        #if (not self.class_weights) and self.use_class_weights:
        #    class_weights = [trial.suggest_float(f'class_weight{i}', 1., 19., step=6.) for i in range(self.num_classes)]
        #else:
        class_weights = self.class_weights


        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
            pin_memory=True,
        )

        batch_factor = 5
        samples_per_epoch = self.batch_size * batch_factor * int(1/(20*batch_factor) * len(val_dataset))
        # 'replacement=False' ensures no duplicates within the same epoch
        sampler = RandomSampler(val_dataset, num_samples=samples_per_epoch, replacement=False)

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size * batch_factor,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            sampler = sampler,
        )

        # Create model with ASPP and residual connections
        self.model = UNet(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_channels=self.base_channels,
            use_residual=self.use_residual,
            use_aspp=self.use_aspp,
            aspp_rates=self.aspp_rates,
            bn_momentum=bn_momentum,
        ).to(self.device)

        logger.info(
            "Model Parameters: %s",
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

        # Loss functions
        if self.use_class_weights:
            class_weights_tensor = torch.tensor(class_weights, device=self.device)
            self.criterion_ce = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            self.criterion_ce = nn.CrossEntropyLoss()

        if self.use_dice_loss:
            self.criterion_dice = DiceLoss(
                smooth=self.dice_loss_smooth, apply_to_classes=self.dice_loss_classes
            )
        else:
            self.criterion_dice = None

        logger.info("Architecture Configuration:")
        logger.info("  Use Residual: %s", self.use_residual)
        logger.info("  Use ASPP: %s", self.use_aspp)
        logger.info("  ASPP rates: %s", self.aspp_rates)
        logger.info("Loss Configuration:")
        if self.use_class_weights:
            logger.info("  Weighted CE class weights: %s", self.class_weights)
        else:
            logger.info("  CE Loss: Vanilla (no class weighting)")
        if self.use_dice_loss:
            logger.info("  Dice loss weight (λ_dice): %.3f", self.dice_loss_weight)
            logger.info("  Dice loss smooth: %.1f", self.dice_loss_smooth)
            logger.info("  Dice loss classes: %s", self.dice_loss_classes or "all")
        else:
            logger.info("  Dice Loss: Disabled")

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=0.0
        )

        # TensorBoard writer
        # self.writer = SummaryWriter(log_dir=str(self.experiment_dir))
        

    def validate_model(self) -> dict[str, float]:
        """Run validation and return loss metrics."""
        self.model.eval()
        total_loss = 0
        total_ce_loss = 0
        total_dice_loss = 0

        with torch.no_grad():
            for inputs, targets in tqdm(
                self.val_loader, desc="Validation", leave=False
            ):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)

                # Compute CE loss
                ce_loss = self.criterion_ce(outputs, targets)
                loss = ce_loss

                # Add Dice loss if enabled
                if self.use_dice_loss:
                    dice_loss = self.criterion_dice(outputs, targets)
                    loss = loss + self.dice_loss_weight * dice_loss
                    total_dice_loss += dice_loss.item()

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()

        avg_loss = total_loss / len(self.val_loader)
        avg_ce_loss = total_ce_loss / len(self.val_loader)

        result = {
            "loss": avg_loss,
            "ce_loss": avg_ce_loss,
        }

        if self.use_dice_loss:
            avg_dice_loss = total_dice_loss / len(self.val_loader)
            result["dice_loss"] = avg_dice_loss

        return result

    def train(self) -> float:
        """Run the step-based training loop."""
        global_step = 0
        validation_cycle = 0
        best_val_loss = float("inf")

        # Training metrics accumulator (since last validation)
        train_running_loss_since_val = 0.0
        train_running_ce_loss_since_val = 0.0
        train_running_dice_loss_since_val = 0.0
        train_samples_since_val = 0

        logger.info("=" * 80)
        logger.info("TRAINING CONFIGURATION")
        logger.info("=" * 80)
        logger.info("Epochs: %d", self.epochs)

        # Architecture info
        arch_components = ["U-Net"]
        if self.use_residual:
            arch_components.append("Residual")
        if self.use_aspp:
            arch_components.append("ASPP")
        logger.info("Architecture: %s", " + ".join(arch_components))
        if self.use_aspp:
            logger.info("ASPP rates: %s", self.aspp_rates)

        # Loss info
        loss_components = []
        if self.use_class_weights:
            loss_components.append(f"Weighted CE {self.class_weights}")
        else:
            loss_components.append("Vanilla CE")
        if self.use_dice_loss:
            loss_components.append(f"Dice (λ={self.dice_loss_weight})")
        logger.info("Loss: %s", " + ".join(loss_components))
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

                # Compute CE loss
                ce_loss = self.criterion_ce(outputs, targets)
                loss = ce_loss

                # Add Dice loss if enabled
                if self.use_dice_loss:
                    dice_loss = self.criterion_dice(outputs, targets)
                    loss = loss + self.dice_loss_weight * dice_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Accumulate training metrics
                batch_size = inputs.size(0)
                train_running_loss_since_val += loss.item() * batch_size
                train_running_ce_loss_since_val += ce_loss.item() * batch_size
                if self.use_dice_loss:
                    train_running_dice_loss_since_val += dice_loss.item() * batch_size
                train_samples_since_val += batch_size

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

                    # Run validation
                    val_metrics = self.validate_model()

                    # Return to training mode
                    self.model.train()

            # Update learning rate at end of epoch
            self.scheduler.step()

        return val_metrics["loss"]

    def __call__(self) -> str:
        """
        Run the HPO (called by submitit).

        Returns
        -------
        str
            optimal Hparams
        """
        logger.info("Starting hpo: %s", self.experiment_id)


        # need to set device on slurm 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_gpus, pruner=pruner)
        
        logger.info("Best params: %s", study.best_params)

        with open(str(self.experiment_dir / "best_params.log"), "w+") as f:
            f.write(str(study.best_params))

        return str(study.best_params) # str(self.checkpoint_dir / "final_model.pth")
