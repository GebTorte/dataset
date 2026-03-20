"""Command-line script to submit LWF-DLR U-Net training to SLURM."""

import argparse
import logging
from pathlib import Path

import submitit

import optuna

#from oma24.training.lwf_unet_aspp_trainer import LWFUNetASPPTrainer
#from oma24.training.lwf_unet_loss_trainer import LWFUNetLossTrainer
#from oma24.training.lwf_unet_skeleton_trainer import LWFUNetSkeletonTrainer
#from src.training.basic_unet_trainer import LWFUNetTrainer
#from src.training.lwf_unet_aspp_trainer import LWFUNetASPPTrainer

from src.tuning.optuna_hpo import LWFUNetASPPOptuna
from src.tuning.optuna_hpo_cs12validate import LWFUNetASPPOptunaCS12Val

logger = logging.getLogger(__name__)


def submit_hpo(
    user: str,
    repo: str = "cloudsen12",
    seed:int = 42,
    csv_name: str = "cloudsen12_initial_cloudfree_high.csv",
    epochs: int = 3,
    batch_size:int = 12,
    num_workers: int = 16,
    prefetch_factor: int = 8,
    n_trials:int = 10,
    experiment_id: str = "cloudsen12_hpo",
    partition: str = "hpda2_compute_gpu",
    time: str = "03:00:00",
    gpus_per_node: int = 1,
    mem_gb: int = 256,
    account: str = "pn39sa-c",
    clusters: str = "hpda2",
    mail_user: str | None = None,
    exclude_nodes: str | None = None,
    use_loss_trainer: bool = False,
    use_aspp_trainer: bool = False,
    use_class_weights: bool = True,
    class_weights: list[float] | None = None,
    use_dice_loss: bool = True,
    dice_loss_weight: float = 0.3,
    use_residual: bool = True,
    use_aspp: bool = True,
    aspp_rates: tuple[int, ...] = (3, 6, 9, 12),
    use_skeleton_trainer: bool = False,
    use_distance: bool = False,
    distance_max: int = 128,
    use_input_skeleton: bool = False,
    use_dual_head: bool = False,
    skeleton_class: int = 1,
    skeleton_loss_weight: float = 1.0,
    cloudsen12_validation: bool = False
) -> submitit.Job:
    """
    Submit LWF-DLR U-Net training job to SLURM cluster.

    Parameters
    ----------
    user : str
        Username for HPC paths.
    csv_name : str
        Name of CSV file with NPZ paths (default: "dev_1000_file.csv").
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    num_workers : int
        Number of data loading workers.
    prefetch_factor : int
        Number of batches to prefetch.
    experiment_id : str
        Unique experiment identifier.
    partition : str
        SLURM partition to use.
    time : str
        Job timeout in "hh:mm:ss" format.
    gpus_per_node : int
        Number of GPUs per node.
    mem_gb : int
        Memory in GB.
    account : str
        SLURM account for billing.
    clusters : str
        SLURM cluster name.
    mail_user : str | None
        Email address for notifications.
    exclude_nodes : str | None
        Comma-separated list of nodes to exclude (e.g., 'i01r04c02s01,i01r04c02s02').
    use_loss_trainer : bool
        Whether to use advanced loss trainer (Weighted CE + Dice).
    use_aspp_trainer : bool
        Whether to use ASPP + Residual trainer (advanced architecture + loss).
    use_class_weights : bool
        Whether to use class weights for CE loss (default: True).
    class_weights : list[float] | None
        Class weights for CE loss (default: [1.0, 50.0, 5.0]).
    use_dice_loss : bool
        Whether to use Dice loss (default: True).
    dice_loss_weight : float
        Weight for Dice loss term (default: 0.3).
    use_residual : bool
        Whether to use residual connections (default: True).
    use_aspp : bool
        Whether to use ASPP in bottleneck (default: True).
    aspp_rates : tuple[int, ...]
        Dilation rates for ASPP (default: (3, 6, 9, 12)).
    use_skeleton_trainer : bool
        Whether to use skeleton-enhanced trainer (default: False).
    use_distance : bool
        Whether to include distance transform as input (default: False).
    distance_max : int
        Maximum distance for normalization (default: 128).
    use_input_skeleton : bool
        Whether to include skeleton as input (default: False).
    use_dual_head : bool
        Whether to use dual-head architecture for skeleton prediction (default: False).
    skeleton_class : int
        Class index to extract skeleton from (default: 1 = linear_vegetation).
    skeleton_loss_weight : float
        Weight for skeleton BCE loss (default: 1.0).

    Returns
    -------
    submitit.Job
        Submitted job object.
    """
    root_hpc = Path("/dss/dsstbyfs02/pn49ci/pn49ci-dss-0026")
    user_path = root_hpc / user
    log_dir = user_path / "experiments/LWF-DLR/slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=str(log_dir))

    cpus_per_task = 2 + num_workers

    slurm_additional_parameters = {
        "clusters": clusters,
        "account": account,
        "get-user-env": True,
        "export": "NONE",
    }

    if mail_user:
        slurm_additional_parameters["mail-type"] = "END"
        slurm_additional_parameters["mail-user"] = mail_user

    if exclude_nodes:
        slurm_additional_parameters["exclude"] = exclude_nodes

    venv_path = root_hpc / user / repo / ".venv"

    executor.update_parameters(
        slurm_partition=partition,
        timeout_min=int(time.split(":")[0]) * 60 + int(time.split(":")[1]),
        gpus_per_node=gpus_per_node,
        cpus_per_task=cpus_per_task,
        mem_gb=mem_gb,
        slurm_job_name=f"hpo_unet_{experiment_id}",
        slurm_additional_parameters=slurm_additional_parameters,
        slurm_setup=[
            "module load slurm_setup",
            f"source {venv_path}/bin/activate",
        ],
    )

    if cloudsen12_validation:
        study = LWFUNetASPPOptunaCS12Val(
            user=user,
            seed=seed,
            train_csv_name=csv_name,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            experiment_id=experiment_id,
            n_trials=n_trials,
            n_gpus=gpus_per_node,
            class_weights=class_weights,
        )
    else:
        study = LWFUNetASPPOptuna(
            user=user,
            seed=seed,
            csv_name=csv_name,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            experiment_id=experiment_id,
            n_trials=n_trials,
            n_gpus=gpus_per_node,
            class_weights=class_weights,
        )

    job = executor.submit(study)

    logger.info("Job submitted with ID: %s", job.job_id)
    logger.info("SLURM log directory: %s", log_dir)
    logger.info("Experiment directory: %s/experiments/LWF-DLR", user_path)


    return job


def main():  # noqa: D103
    parser = argparse.ArgumentParser(
        description="Submit LWF-DLR U-Net segmentation hpo to SLURM cluster"
    )

    # Required arguments
    parser.add_argument(
        "--user",
        type=str,
        required=True,
        help="Username for HPC paths (e.g., di38gaz)",
    )

    # Data configuration
    parser.add_argument(
        "--csv-name",
        type=str,
        default="cloudsen12_initial_cloudfree_high.csv",
        help='CSV file with NPZ/TIF paths (default: "cloudsen12_initial_cloudfree_high.csv")',
    )

    # hpo params
    parser.add_argument(
        "--n-trials", type=int, default=5, help="n trials for hpo optuna (default 5)"
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of hpo training epochs per model (default: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42, #consider setting to None, so default mode will be random
        help="Random seed (int) to pass onto torch and numpy"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Batch size for training (default: 12)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of data loading workers (default: 16)",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=8,
        help="Number of batches to prefetch (default: 8)",
    )

    # Experiment configuration
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="cloudsen12_aspp_hpo",
        help="Unique experiment identifier (default: baseline_unet_slurm)",
    )

    # SLURM configuration
    parser.add_argument(
        "--partition",
        type=str,
        default="hpda2_compute_gpu",
        help="SLURM partition (default: hpda2_compute_gpu)",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="24:00:00",
        help='Job timeout in "hh:mm:ss" format (default: 24:00:00)',
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=1,
        help="Number of GPUs per node (default: 1)",
    )
    parser.add_argument(
        "--mem-gb", type=int, default=256, help="Memory in GB (default: 256)"
    )
    parser.add_argument(
        "--account",
        type=str,
        default="pn39sa-c",
        help="SLURM account for billing (default: pn39sa-c)",
    )
    parser.add_argument(
        "--clusters",
        type=str,
        default="hpda2",
        help="SLURM cluster name (default: hpda2)",
    )
    parser.add_argument(
        "--mail-user", type=str, default=None, help="Email for job notifications"
    )
    parser.add_argument(
        "--exclude-nodes",
        type=str,
        default=None,
        help=(
            "Comma-separated list of nodes to exclude "
            "(e.g., 'i01r04c02s01,i01r04c02s02')"
        ),
    )

    # Loss function configuration
    parser.add_argument(
        "--loss",
        action="store_true",
        help="Use advanced loss trainer (Weighted CE + Dice)",
    )
    parser.add_argument(
        "--aspp",
        action="store_true",
        help="Use ASPP + Residual trainer (advanced architecture + loss)",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weighting for CE loss (use vanilla CE)",
    )
    parser.add_argument(
        "--class-weights",
        type=float,
        nargs=4,
        default=None,
        help=(
            "Class weights for CE loss "
            "(example: [1.0, 15.0, 10.0, 5.0] for clear, cloud, thin cloud, shadow"
        ),
    )
    parser.add_argument(
        "--no-dice-loss",
        action="store_true",
        help="Disable Dice loss (use only CE loss)",
    )
    parser.add_argument(
        "--dice-loss-weight",
        type=float,
        default=0.3,
        help="Weight for Dice loss term (default: 0.3)",
    )

    # Architecture configuration
    parser.add_argument(
        "--no-residual",
        action="store_true",
        help="Disable residual connections (default: enabled)",
    )
    parser.add_argument(
        "--no-aspp-bottleneck",
        action="store_true",
        help="Disable ASPP in bottleneck (default: enabled)",
    )
    parser.add_argument(
        "--aspp-rates",
        type=int,
        nargs="+",
        default=[3, 6, 9, 12],
        help="Dilation rates for ASPP (default: 3 6 9 12)",
    )

    # Skeleton enhancement configuration
    parser.add_argument(
        "--skeleton",
        action="store_true",
        help="Use skeleton-enhanced trainer with optional features",
    )
    parser.add_argument(
        "--distance",
        action="store_true",
        help="Include distance transform as input channel",
    )
    parser.add_argument(
        "--distance-max",
        type=int,
        default=128,
        help="Maximum distance for normalization (default: 128)",
    )
    parser.add_argument(
        "--input-skeleton",
        action="store_true",
        help="Include skeleton as input channel",
    )
    parser.add_argument(
        "--dual-head",
        action="store_true",
        help="Use dual-head architecture for skeleton prediction",
    )
    parser.add_argument(
        "--skeleton-class",
        type=int,
        default=1,
        help="Class index to extract skeleton from (default: 1 = linear_vegetation)",
    )
    parser.add_argument(
        "--skeleton-loss-weight",
        type=float,
        default=1.0,
        help="Weight for skeleton BCE loss (default: 1.0)",
    )

    parser.add_argument(
        "--use-cloudsen12-validation",
        action="store_true",
        help="whether to use cloudsen12 data as validation during training."
    )

    args = parser.parse_args()

    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Submit the job
    job = submit_hpo(
        user=args.user,
        seed=args.seed,
        csv_name=args.csv_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        n_trials=args.n_trials,
        experiment_id=args.experiment_id,
        partition=args.partition,
        time=args.time,
        gpus_per_node=args.gpus_per_node,
        mem_gb=args.mem_gb,
        account=args.account,
        clusters=args.clusters,
        mail_user=args.mail_user,
        exclude_nodes=args.exclude_nodes,
        use_loss_trainer=args.loss,
        use_aspp_trainer=args.aspp,
        use_class_weights=not args.no_class_weights,
        class_weights=args.class_weights,
        use_dice_loss=not args.no_dice_loss,
        dice_loss_weight=args.dice_loss_weight,
        use_residual=not args.no_residual,
        use_aspp=not args.no_aspp_bottleneck,
        aspp_rates=tuple(args.aspp_rates),
        use_skeleton_trainer=args.skeleton,
        use_distance=args.distance,
        distance_max=args.distance_max,
        use_input_skeleton=args.input_skeleton,
        use_dual_head=args.dual_head,
        skeleton_class=args.skeleton_class,
        skeleton_loss_weight=args.skeleton_loss_weight,
        cloudsen12_validation=args.use_cloudsen12_validation,
    )

    logger.info("Job submitted successfully!")
    logger.info("Job ID: %s", job.job_id)
    logger.info("To check job status: squeue -j %s", job.job_id)
    logger.info("To cancel job: scancel %s", job.job_id)



if __name__ == "__main__":
    main()