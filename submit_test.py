"""Command-line script to submit LWF-DLR U-Net validation to SLURM."""

import argparse
import logging
from pathlib import Path

import submitit

from src.dataset import TestS2TIFDataSet, TestS2TIFDataSet512

import torch

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np

from dataset import TestS2Dataset


class DiceLoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.softmax(logits, dim=1)

        targets_onehot = F.one_hot(targets, self.num_classes)
        targets_onehot = targets_onehot.permute(0,3,1,2).float()

        dims = (0,2,3)

        intersection = torch.sum(probs * targets_onehot, dims)
        union = torch.sum(probs + targets_onehot, dims)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


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
        model_path,
        tif_paths,
        batch_size=12,
        num_workers=4,
        device="cuda",
        experiment_name="test_run"
    ):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")


        #checkpoint = torch.load("checkpoint.pth")

        #model.load_state_dict(checkpoint["model_state_dict"])
        #optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        #epoch = checkpoint["epoch"]

        #model.eval()

        self.model = model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        dataset = TestS2Dataset(tif_paths)

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

        self.num_classes = 4

        self.exp_dir = Path("experiments") / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.exp_dir / "tensorboard")

    def run(self):

        total_ce = 0
        total_dice_loss = 0

        total_pixels = 0
        correct_pixels = 0

        dice_scores = np.zeros(self.num_classes)
        iou_scores = np.zeros(self.num_classes)

        num_images = 0

        with torch.no_grad():

            for step, (images, masks) in enumerate(self.loader):

                images = images.to(self.device)
                masks = masks.to(self.device)

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
                self.writer.add_scalar("Batch/CrossEntropy", ce.item(), step)
                self.writer.add_scalar("Batch/DiceLoss", dl.item(), step)

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
        with open(self.exp_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print(json.dumps(metrics, indent=4))

        return metrics