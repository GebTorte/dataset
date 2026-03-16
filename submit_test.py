"""Command-line script to submit LWF-DLR U-Net validation to SLURM."""

import argparse
import logging
from pathlib import Path

import submitit

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from tqdm.auto import tqdm

from src.misc import select_patches_from_dataset
from src.dataset import TestS2TIFDataSet, TestS2TIFDataSet512
from src.model import UNet
from src.training.lwf_unet_aspp_trainer import UNet as ASPPUNet

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks.

    Dice coefficient measures overlap between prediction and ground truth.
    Dice Loss = 1 - Dice coefficient

    Args:
        smooth: Smoothing factor to avoid division by zero (default: 1.0)
        apply_to_classes: List of class indices to compute loss for (None = all classes)
    """

    def __init__(self, smooth=1.0, apply_to_classes=None):
        super().__init__()
        self.smooth = smooth
        self.apply_to_classes = apply_to_classes

    def forward(self, logits, targets):
        """
        Args:
            logits: Model output (B, C, H, W) - raw scores before softmax
            targets: Ground truth (B, H, W) - class indices

        Returns:
            Dice loss (scalar)
        """
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1)  # (B, C, H, W)

        num_classes = probs.shape[1]

        # One-hot encode targets: (B, H, W) -> (B, C, H, W)
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        # Determine which classes to compute loss for
        if self.apply_to_classes is not None:
            class_indices = self.apply_to_classes
        else:
            class_indices = range(num_classes)

        dice_loss = 0.0
        num_classes_used = 0

        for c in class_indices:
            # Get predictions and targets for class c
            pred_c = probs[:, c, :, :]  # (B, H, W)
            target_c = targets_one_hot[:, c, :, :]  # (B, H, W)

            # Flatten spatial dimensions
            pred_c = pred_c.reshape(pred_c.shape[0], -1)  # (B, H*W)
            target_c = target_c.reshape(target_c.shape[0], -1)  # (B, H*W)

            # Compute Dice coefficient
            intersection = (pred_c * target_c).sum(dim=1)  # (B,)
            union = pred_c.sum(dim=1) + target_c.sum(dim=1)  # (B,)

            dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1.0 - dice_coeff).mean()
            num_classes_used += 1

        # Average over classes
        return dice_loss / num_classes_used

def dice_per_class(pred, target, num_classes=4):

    dice_scores = []

    for c in range(num_classes):

        pred_c = (pred == c)
        target_c = (target == c)

        intersection = (pred_c & target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2 * intersection) / (union + 1e-6)

        dice_scores.append(dice)

    return dice_scores


def iou_per_class(pred, target, num_classes=4):

    ious = []

    for c in range(num_classes):

        pred_c = (pred == c)
        target_c = (target == c)

        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()

        iou = intersection / (union + 1e-6)

        ious.append(iou)

    return ious


class ModelTester:

    def __init__(
        self,
        model,
        dataset,
        batch_size=12,
        num_workers=16,
        device="cuda",
        user:str = "di54xat",
        repo:str = "cloudsen12",
        experiment_group: str = "LWF-DLR",
        experiment_id="test_run",
        model_name="best_model.pth",
        root_hpc: Path|None = None,
    ):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.root_hpc = root_hpc or Path("/dss/dsstbyfs02/pn49ci/pn49ci-dss-0026")
        self.user_path = self.root_hpc / user
        self.data_path = self.user_path / repo / "data"
        self.data_root = self.data_path

        self.experiment_group = experiment_group
        self.experiment_id = experiment_id

        self.experiment_dir = self.user_path / "experiments" / experiment_group / experiment_id
     
        self.model_dir = self.experiment_dir / "checkpoints"

        #model.load_state_dict(checkpoint["model_state_dict"])
        #optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        #epoch = checkpoint["epoch"]
        #model.eval()

        save_dict = torch.load(self.model_dir / model_name)

        self.model = model.to(self.device)

        self.model.load_state_dict(save_dict["model_state_dict"])
        self.model.eval()

        self.dataset = dataset

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

        self.num_classes = 4

        self.writer = SummaryWriter(log_dir=self.experiment_dir / "tensorboard/test")

    def __call__(self) ->  str:
        """
        Run the main test loop (called by submitit).

        Returns
        -------
        str
            Path to the metrics directory
        """
        logger.info("Starting test: %s", self.experiment_id)

        metrics = self.test()

        return json.dumps(metrics, indent=4)


    def test(self):

        total_ce = 0
        total_dice_loss = 0

        total_pixels = 0
        correct_pixels = 0

        dice_scores = np.zeros(self.num_classes)
        iou_scores = np.zeros(self.num_classes)

        num_images = 0

        global_step = 0


        with torch.no_grad():
            
            for batch_idx, (inputs, targets) in enumerate(
                tqdm(self.loader, desc=f"Test")
            ):
            #for step, (images, masks) in enumerate(self.loader):

                images = inputs.to(self.device)
                masks = targets.to(self.device)

                logits = self.model(images)

                ce = self.ce_loss(logits, masks)
                dl = self.dice_loss(logits, masks)

                total_ce += ce.item()
                total_dice_loss += dl.item()

                preds = torch.argmax(logits, dim=1)

                correct_pixels += (preds == masks).sum().item()
                total_pixels += torch.numel(masks)

                preds_np = preds.cpu().numpy()
                masks_np = masks.cpu().numpy()

                for p, t in zip(preds_np, masks_np):

                    d = dice_per_class(p, t, self.num_classes)
                    i = iou_per_class(p, t, self.num_classes)

                    dice_scores += np.array(d)
                    iou_scores += np.array(i)

                    num_images += 1

                # batch metrics to tensorboard
                self.writer.add_scalar("Batch/CrossEntropy", ce.item(), global_step)
                self.writer.add_scalar("Batch/DiceLoss", dl.item(), global_step)

        dice_scores /= num_images
        iou_scores /= num_images

        pixel_acc = correct_pixels / total_pixels

        metrics = {
            "cross_entropy": total_ce / len(self.loader),
            "dice_loss": total_dice_loss / len(self.loader),
            "pixel_accuracy": pixel_acc,
            "dice_per_class": dice_scores.tolist(),
            "mean_dice": float(dice_scores.mean()),
            "iou_per_class": iou_scores.tolist(),
            "mean_iou": float(iou_scores.mean())
        }

        # log final metrics
        self.writer.add_scalar("Test/CrossEntropy", metrics["cross_entropy"])
        self.writer.add_scalar("Test/DiceLoss", metrics["dice_loss"])
        self.writer.add_scalar("Test/PixelAccuracy", metrics["pixel_accuracy"])
        self.writer.add_scalar("Test/MeanDice", metrics["mean_dice"])
        self.writer.add_scalar("Test/MeanIoU", metrics["mean_iou"])

        for i in range(self.num_classes):

            self.writer.add_scalar(f"Dice/Class_{i}", dice_scores[i])
            self.writer.add_scalar(f"IoU/Class_{i}", iou_scores[i])

        self.writer.close()

        # save metrics to json
        with open(self.experiment_dir / "test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        return metrics


def submit_slurm_testing(
    user: str,
    repo: str = "cloudsen12",
    seed:int = 42,
    csv_name: str = "cloudsen12_initial_cloudfree_dev_200.csv",
    epochs: int = 16,
    patch_size: int = 256,
    batch_size: int = 12,
    num_workers: int = 16,
    prefetch_factor: int = 8,
    experiment_id: str = "cloudsen12_aspp_scribble_multitypecloud_256_001",
    partition: str = "hpda2_compute_gpu",
    time: str = "03:00:00",
    gpus_per_node: int = 1,
    mem_gb: int = 256,
    account: str = "pn39sa-c",
    clusters: str = "hpda2",
    mail_user: str | None = None,
    exclude_nodes: str | None = None,
) -> submitit.Job:

    root_hpc = Path("/dss/dsstbyfs02/pn49ci/pn49ci-dss-0026")
    user_path = root_hpc / user
    log_dir = user_path / "experiments/LWF-DLR/slurm_logs/test"
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
        slurm_job_name=f"lwf_unet_{experiment_id}",
        slurm_additional_parameters=slurm_additional_parameters,
        slurm_setup=[
            "module load slurm_setup",
            f"source {venv_path}/bin/activate",
        ],
    )

    experiment_group: str = "LWF-DLR"

    data_path = user_path / repo / "data"

    # Load dataset
    file_names = select_patches_from_dataset(
        data_path / csv_name, 
        data_path,
        type_folder="", # not supported here
    )


    if use_aspp_trainer:
        model = ASPPUNet()
    else:
        model = UNet()


    if patch_size == 512:
        dataset = TestS2TIFDataSet512(
            file_names,
            seed
        )
    else: #default 256
        dataset = TestS2TIFDataSet(
            file_names,
            seed
        )

    model_name = "best_model.pth"
    tester = ModelTester(
        model,
        dataset,
        batch_size,
        num_workers,
        user,
        experiment_group, 
        experiment_id,
        model_name,
    )

    job = executor.submit(tester)

    logger.info("Job submitted with ID: %s", job.job_id)
    logger.info("SLURM log directory: %s", log_dir)
    logger.info("Experiment directory: %s/experiments/LWF-DLR", user_path)

    return job


def main():
    parser = argparse.ArgumentParser(
        description="Submit LWF-DLR U-Net segmentation testing to SLURM cluster"
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

    # parameters
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Defining the Test data loader (image size). One of [256, 512]."
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
        default="cloudsen12_baseline_unet_slurm",
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

    args = parser.parse_args()

    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Submit the job
    job = submit_slurm_testing(
        user=args.user,
        seed=args.seed,
        csv_name=args.csv_name,
        epochs=args.epochs,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        experiment_id=args.experiment_id,
        partition=args.partition,
        time=args.time,
        gpus_per_node=args.gpus_per_node,
        mem_gb=args.mem_gb,
        account=args.account,
        clusters=args.clusters,
        mail_user=args.mail_user,
        exclude_nodes=args.exclude_nodes,
    )

    logger.info("Job submitted successfully!")
    logger.info("Job ID: %s", job.job_id)
    logger.info("To check job status: squeue -j %s", job.job_id)
    logger.info("To cancel job: scancel %s", job.job_id)




if __name__ == "__main__":
    main()
