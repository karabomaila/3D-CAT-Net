#!/usr/bin/env python
"""
For Training
Extended from CAT-Net code by Yi Lin et al.
"""

import logging
import os
import random
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from config import ex
from dataloaders.load_dataset import TrainDataset as TrainDataset
from models.fewshot import FewShotSeg
from utils.utils import *


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        # Set up source folder
        os.makedirs(f"{_run.observers[0].dir}/snapshots", exist_ok=True)
        for source_file, _ in _run.experiment_info["sources"]:
            os.makedirs(
                os.path.dirname(f"{_run.observers[0].dir}/source/{source_file}"),
                exist_ok=True,
            )
            _run.observers[0].save_file(source_file, f"source/{source_file}")
        shutil.rmtree(f"{_run.observers[0].basedir}/_sources")

        # Set up logger
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
    model.train()

    _log.info("Set optimizer...")
    optimizer = torch.optim.SGD(model.parameters(), **_config["optim"])

    lr_milestones = [
        (i + 1) * _config["max_iters_per_load"]
        for i in range(_config["n_steps"] // _config["max_iters_per_load"] - 1)
    ]
    scheduler = MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=_config["lr_step_gamma"]
    )

    my_weight = torch.FloatTensor([0.1, 1.0]).to(device)
    criterion = nn.NLLLoss(ignore_index=255, weight=my_weight)

    _log.info("Load data...")
    data_config = {
        "data_dir": "./data/CHAOST2/niis/T2SPIR",
        "dataset": _config["dataset"],
        "n_shot": _config["n_shot"],
        "n_way": _config["n_way"],
        "n_query": _config["n_query"],
        "max_iter": _config["max_iters_per_load"],
        "eval_fold": _config["eval_fold"],
        "test_label": _config["test_label"],
        "exclude_label": _config["exclude_label"],
    }
    train_dataset = TrainDataset(data_config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=_config["batch_size"],
        shuffle=True,
        num_workers=_config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    n_sub_epochs = (
        _config["n_steps"] // _config["max_iters_per_load"]
    )  # number of times for reloading

    log_loss: dict[str, int] = {"total_loss": 0, "query_loss": 0, "align_loss": 0}

    i_iter = 0
    _log.info("Start training...")
    for sub_epoch in range(n_sub_epochs):
        _log.info(f'This is epoch "{sub_epoch}" of "{n_sub_epochs}" epochs.')
        for _, sample in enumerate(train_loader):
            # Prepare episode data.
            support_images = [
                [shot.float().to(device) for shot in way]
                for way in sample["support_images"]
            ]

            support_fg_mask = [
                [shot.float().to(device) for shot in way]
                for way in sample["support_fg_labels"]
            ]

            query_images = [
                query_image.float().to(device) for query_image in sample["query_images"]
            ]

            query_labels = torch.cat(
                [
                    query_label.long().to(device)
                    for query_label in sample["query_labels"]
                ],
                dim=0,
            )

            # Compute outputs and losses.
            query_pred, align_loss = model(
                support_images, support_fg_mask, query_images, train=True
            )

            query_loss = criterion(
                torch.log(
                    torch.clamp(
                        query_pred,
                        torch.finfo(torch.float32).eps,
                        1 - torch.finfo(torch.float32).eps,
                    )
                ),
                query_labels,
            )

            query_pred = query_pred.argmax(dim=1).cpu()

            loss = query_loss + align_loss

            # Compute gradient and do SGD step.
            for param in model.parameters():
                param.grad = None

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            query_loss = query_loss.detach().data.cpu().numpy()
            align_loss = align_loss.detach().data.cpu().numpy()

            _run.log_scalar("total_loss", loss.item())
            _run.log_scalar("query_loss", query_loss)
            _run.log_scalar("align_loss", align_loss)

            log_loss["total_loss"] += loss.item()
            log_loss["query_loss"] += query_loss
            log_loss["align_loss"] += align_loss

            # Print loss and take snapshots.
            if (i_iter) % _config["print_interval"] == 0:
                total_loss = log_loss["total_loss"] / _config["print_interval"]
                query_loss = log_loss["query_loss"] / _config["print_interval"]
                align_loss = log_loss["align_loss"] / _config["print_interval"]

                log_loss["total_loss"] = 0
                log_loss["query_loss"] = 0
                log_loss["align_loss"] = 0

                _log.info(
                    f"step {i_iter}: total_loss: {total_loss}, query_loss: {query_loss},"
                    f" align_loss: {align_loss}"
                )

            if (i_iter + 1) % _config["save_snapshot_every"] == 0:
                _log.info("###### Taking snapshot ######")
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        f"{_run.observers[0].dir}/snapshots", f"{i_iter + 1}.pth"
                    ),
                )

            i_iter += 1

        if (sub_epoch + 1) == n_sub_epochs:
            _log.info("###### Taking snapshot ######")
            torch.save(
                model.state_dict(),
                os.path.join(f"{_run.observers[0].dir}/snapshots", f"{i_iter + 1}.pth"),
            )

    _log.info("End of training.")
    return 1
