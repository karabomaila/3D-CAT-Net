#!/usr/bin/env python
"""
For evaluation
Extended from CAT-Net code by Yi Lin et al.
"""

import logging
import os
import shutil

import numpy as np
import SimpleITK as sitk
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader

from config import ex
from dataloaders.dataset_specifics import *
from dataloaders.load_dataset import TestDataset
from models.fewshot import FewShotSeg
from utils import *


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f"{_run.observers[0].dir}/interm_preds", exist_ok=True)
        for source_file, _ in _run.experiment_info["sources"]:
            os.makedirs(
                os.path.dirname(f"{_run.observers[0].dir}/source/{source_file}"),
                exist_ok=True,
            )
            _run.observers[0].save_file(source_file, f"source/{source_file}")
        shutil.rmtree(f"{_run.observers[0].basedir}/_sources")

        # Set up logger -> log to .txt
        file_handler = logging.FileHandler(
            os.path.join(f"{_run.observers[0].dir}", "logger.log")
        )
        file_handler.setLevel("INFO")
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        _log.handlers.append(file_handler)
        _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

    # Deterministic setting for reproducibility.
    if _config["seed"] is not None:
        random.seed(_config["seed"])
        torch.manual_seed(_config["seed"])
        torch.cuda.manual_seed_all(_config["seed"])
        cudnn.deterministic = True

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _log.info("Create model...")
    model: FewShotSeg = FewShotSeg(_config["opt"]).to(device)
    model.load_state_dict(torch.load(_config["reload_model_path"], map_location="cpu"))

    _log.info("Load data...")
    data_config = {
        "data_dir": "./data/CHAOST2/niis/T2SPIR",
        "dataset": _config["dataset"],
        "n_shot": _config["n_shot"],
        "n_way": _config["n_way"],
        "n_query": _config["n_query"],
        "n_sv": _config["n_sv"],
        "max_iter": _config["max_iters_per_load"],
        "eval_fold": _config["eval_fold"],
        "min_size": _config["min_size"],
        "max_slices": _config["max_slices"],
        "supp_idx": _config["supp_idx"],
    }
    test_dataset = TestDataset(data_config)
    test_loader = DataLoader(
        test_dataset,
        batch_size=_config["batch_size"],
        shuffle=False,
        num_workers=_config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # Get unique labels (classes).
    labels = get_label_names(_config["dataset"])

    # Loop over classes.
    class_dice = {}
    class_iou = {}

    _log.info("Starting validation...")
    for label_val, label_name in labels.items():
        # Skip BG class.
        if label_name == "BG":
            continue
        elif not np.intersect1d([label_val], _config["test_label"]).size:
            continue

        _log.info(f"Test Class: {label_name}")

        # Get support sample + mask for current class.
        support_sample = test_dataset.getSupport(label=label_val, N=_config["n_part"])
        test_dataset.label = label_val

        # Test.
        with torch.no_grad():
            model.eval()

            support_images = [
                [shot.float().to(device) for shot in way]
                for way in support_sample["images"]
            ]

            support_fg_mask = [
                [shot.float().to(device) for shot in way]
                for way in support_sample["labels"]
            ]

            # Loop through query volumes.
            scores = Scores()
            for i, sample in enumerate(test_loader):
                # [C x 3 x H x W]
                query_image = [
                    query_image.float().to(device) for query_image in sample["image"]
                ]

                query_label = torch.cat(
                    [query_label.long().to(device) for query_label in sample["label"]],
                    dim=0,
                )

                query_id = sample["id"][0].split("image_")[1][: -len(".nii.gz")]

                # Compute output.
                query_pred, align_loss = model(
                    support_images,
                    support_fg_mask,
                    query_image,
                    train=False,
                    n_iters=_config["n_iters"],
                )

                query_pred = query_pred.argmax(dim=1).cpu()

                # Record scores.
                scores.record(query_pred, query_label)
                # assert False

                # Log.
                _log.info(
                    f'Tested query volume: {sample["id"][0][len(_config["path"][_config["dataset"]]["data_dir"]):]}.'
                )
                _log.info(f"Dice score: {scores.patient_dice[-1].item()}")

                # Save predictions.
                file_name = os.path.join(
                    f"{_run.observers[0].dir}/interm_preds",
                    f"prediction_{query_id}_{label_name}.nii.gz",
                )
                itk_pred = sitk.GetImageFromArray(query_pred[0])
                sitk.WriteImage(itk_pred, file_name, True)
                _log.info(f"{query_id} has been saved. ")

            # Log class-wise results
            class_dice[label_name] = torch.tensor(scores.patient_dice).mean().item()
            class_iou[label_name] = torch.tensor(scores.patient_iou).mean().item()
            _log.info(f"Test Class: {label_name}")
            _log.info(f"Mean class IoU: {class_iou[label_name]}")
            _log.info(f"Mean class Dice: {class_dice[label_name]}")

    _log.info("Final results...")
    _log.info(f"Mean IoU: {class_iou}")
    _log.info(f"Mean Dice: {class_dice}")

    _log.info("End of validation.")
    return 1
