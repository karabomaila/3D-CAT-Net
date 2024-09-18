"""
Dataset for Training and Test
Extended from CAT-Net code by Yi Lin et al.
"""

import glob
import os
import random

import numpy as np
import SimpleITK as sitk
import torch
import torchvision.transforms as deftfx
from torch.utils.data import Dataset

from . import image_transforms as myit
from .dataset_specifics import *

images_path = r"./data/CHAOST2/niis/T2SPIR/normalized/image*"
label_images_path = r"./data/CHAOST2/niis/T2SPIR/normalized/label*"


class TestDataset(Dataset):
    def __init__(self, args) -> None:
        # reading the paths
        if args["dataset"] == "CMR":
            self.image_dirs = glob.glob(
                os.path.join(args["data_dir"], "cmr_MR_normalized/image*")
            )
        elif args["dataset"] == "CHAOST2":
            self.image_dirs = glob.glob(images_path)
        elif args["dataset"] == "SABS":
            self.image_dirs = glob.glob(
                os.path.join(args["data_dir"], "sabs_CT_normalized/image*")
            )

        self.image_dirs: list[str] = sorted(
            self.image_dirs, key=lambda x: int(x.split("_")[-1].split(".nii.gz")[0])
        )

        # remove test fold!
        self.FOLD = get_folds(args["dataset"])
        self.image_dirs = [
            elem
            for idx, elem in enumerate(self.image_dirs)
            if idx in self.FOLD[args["eval_fold"]]
        ]

        # split into support/query
        idx = np.arange(len(self.image_dirs))
        self.support_dir = self.image_dirs[idx[args["supp_idx"]]]
        self.image_dirs.pop(idx[args["supp_idx"]])  # remove support
        self.label = None

    def __len__(self):
        return len(self.image_dirs)

    def resize_image(self, image, new_size, interpolator=sitk.sitkLinear):
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        original_direction = image.GetDirection()
        original_origin = image.GetOrigin()

        # Calculate new spacing based on the new size
        new_spacing = [
            original_size[i] * original_spacing[i] / new_size[i]
            for i in range(len(original_size))
        ]

        resample = sitk.ResampleImageFilter()
        resample.SetSize(new_size)
        resample.SetOutputSpacing(new_spacing)
        resample.SetOutputDirection(original_direction)
        resample.SetOutputOrigin(original_origin)
        resample.SetInterpolator(interpolator)
        resample.SetDefaultPixelValue(image.GetPixelIDValue())

        return resample.Execute(image)

    def __getitem__(self, idx):
        img_path = self.image_dirs[idx]

        img = self.resize_image(sitk.ReadImage(img_path), new_size=[256, 256, 31])
        img = sitk.GetArrayFromImage(img)
        img = (img - img.mean()) / img.std()
        img = np.stack(([img], [img], [img]), axis=1)

        lbl = self.resize_image(
            sitk.ReadImage(
                img_path.split("image_")[0] + "label_" + img_path.split("image_")[-1]
            ),
            new_size=[256, 256, 31],
        )
        lbl = sitk.GetArrayFromImage(lbl)

        # lbl[lbl == 200] = 1
        # lbl[lbl == 500] = 2
        # lbl[lbl == 600] = 3
        lbl = 1 * (lbl == self.label)

        sample = {"id": img_path}

        sample["image"] = torch.from_numpy(img)
        sample["label"] = torch.from_numpy(lbl)

        return sample

    def get_support_index(self, n_shot, C):
        """
        Selecting intervals according to Ouyang et al.
        """
        if n_shot == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (n_shot * 2)
            part_interval = (1.0 - 1.0 / n_shot) / (n_shot - 1)
            pcts = [half_part + part_interval * ii for ii in range(n_shot)]

        return (np.array(pcts) * C).astype("int")

    def getSupport(self, label=None, all_slices=True, N=None):
        if label is None:
            raise ValueError("Need to specify label class!")

        img_path = self.support_dir
        img = self.resize_image(sitk.ReadImage(img_path), new_size=[256, 256, 31])
        img = sitk.GetArrayFromImage(img)
        img = (img - img.mean()) / img.std()

        img = [[img]]
        img = np.stack((img, img, img), axis=2)

        lbl = self.resize_image(
            sitk.ReadImage(
                img_path.split("image_")[0] + "label_" + img_path.split("image_")[-1]
            ),
            new_size=[256, 256, 31],
        )

        lbl = sitk.GetArrayFromImage(lbl)
        lbl = lbl[np.newaxis, ...]
        # lbl[lbl == 200] = 1
        # lbl[lbl == 500] = 2
        # lbl[lbl == 600] = 3

        lbl = 1 * (lbl == label)

        sample = {}
        sample["images"] = [torch.from_numpy(img)]
        sample["labels"] = [[torch.from_numpy(lbl)]]

        return sample


class TrainDataset(Dataset):
    def __init__(self, args):
        self.n_shot = args["n_shot"]
        self.n_way = args["n_way"]
        self.n_query = args["n_query"]
        self.max_iter = args["max_iter"]
        self.read = True  # read images before get_item
        self.train_sampling = "neighbors"
        self.test_label = args["test_label"]
        self.exclude_label = args["exclude_label"]

        # reading the paths (leaving the reading of images into memory to __getitem__)
        if args["dataset"] == "CMR":
            self.image_dirs = glob.glob(
                os.path.join(args["data_dir"], "cmr_MR_normalized/image*")
            )
            self.label_dirs = glob.glob(
                os.path.join(args["data_dir"], "cmr_MR_normalized/label*")
            )
        elif args["dataset"] == "CHAOST2":
            self.image_dirs = glob.glob(images_path)
            self.label_dirs = glob.glob(label_images_path)
        elif args["dataset"] == "SABS":
            self.image_dirs = glob.glob(
                os.path.join(args["data_dir"], "sabs_CT_normalized/image*")
            )
            self.label_dirs = glob.glob(
                os.path.join(args["data_dir"], "sabs_CT_normalized/label*")
            )

        self.image_dirs = sorted(
            self.image_dirs, key=lambda x: int(x.split("_")[-1].split(".nii.gz")[0])
        )
        self.label_dirs = sorted(
            self.label_dirs, key=lambda x: int(x.split("_")[-1].split(".nii.gz")[0])
        )

        # remove test fold!
        self.FOLD = get_folds(args["dataset"])
        self.image_dirs = [
            elem
            for idx, elem in enumerate(self.image_dirs)
            if idx not in self.FOLD[args["eval_fold"]]
        ]
        self.label_dirs = [
            elem
            for idx, elem in enumerate(self.label_dirs)
            if idx not in self.FOLD[args["eval_fold"]]
        ]

        # read images
        if self.read:
            self.support_images = []
            self.support_labels = []

            self.query_images = []
            self.query_labels = []
            id = 0
            for image_dir, label_dir in zip(self.image_dirs, self.label_dirs):
                resize_image = self.resize_image(
                    sitk.ReadImage(image_dir), new_size=[256, 256, 31]
                )
                resize_mask = self.resize_image(
                    sitk.ReadImage(label_dir), new_size=[256, 256, 31]
                )

                image = sitk.GetArrayFromImage(resize_image)
                mask = sitk.GetArrayFromImage(resize_mask)

                if id in [12, 16]:
                    self.support_images.append(image)
                    self.support_labels.append(mask)
                else:
                    self.query_images.append(image)
                    self.query_labels.append(mask)
                id += 1

    def __len__(self):
        return self.max_iter

    def gamma_tansform(self, img):
        gamma_range = (0.5, 1.5)
        gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
        cmin = img.min()
        irange = img.max() - cmin + 1e-5

        img = img - cmin + 1e-5
        img = irange * np.power(img * 1.0 / irange, gamma)
        img = img + cmin

        return img

    def geom_transform(self, img, mask):
        affine = {"rotate": 5, "shift": (5, 5), "shear": 5, "scale": (0.9, 1.2)}
        alpha = 10
        sigma = 5
        order = 3

        tfx = []
        tfx.append(
            myit.RandomAffine(
                affine.get("rotate"),
                affine.get("shift"),
                affine.get("shear"),
                affine.get("scale"),
                affine.get("scale_iso", True),
                order=order,
            )
        )
        tfx.append(myit.ElasticTransform(alpha, sigma))
        transform = deftfx.Compose(tfx)

        if len(img.shape) > 5:
            n_shot = img.shape[1]
            for shot in range(n_shot):
                cat = np.concatenate((img[0, shot], mask[:, shot])).transpose(
                    1, 2, 3, 0
                )
                cat = transform(cat).transpose(3, 0, 1, 2)

                img[0, shot] = cat[:3, :, :]
                mask[:, shot] = np.rint(cat[3:, :, :])

        else:
            for q in range(img.shape[0]):
                cat = np.concatenate((img[q], mask[q][None])).transpose(1, 2, 3, 0)
                cat = transform(cat).transpose(3, 0, 1, 2)
                img[q] = cat[:3, :, :]
                mask[q] = np.rint(cat[3:, :, :].squeeze())

        return img, mask

    def resize_image(self, image, new_size, interpolator=sitk.sitkLinear):
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        original_direction = image.GetDirection()
        original_origin = image.GetOrigin()

        # Calculate new spacing based on the new size
        new_spacing = [
            original_size[i] * original_spacing[i] / new_size[i]
            for i in range(len(original_size))
        ]

        resample = sitk.ResampleImageFilter()
        resample.SetSize(new_size)
        resample.SetOutputSpacing(new_spacing)
        resample.SetOutputDirection(original_direction)
        resample.SetOutputOrigin(original_origin)
        resample.SetInterpolator(interpolator)
        resample.SetDefaultPixelValue(image.GetPixelIDValue())

        return resample.Execute(image)

    def __getitem__(self, idx):
        support_idx: int = random.choice(range(len(self.support_images)))
        pat_idx: int = random.choice(range(len(self.query_images)))

        train_labels: list[int] = [1, 2, 3, 4]
        train_labels = list(set(train_labels) - set(self.test_label))
        label: int = random.choice(train_labels)

        if self.read:
            support_img = self.support_images[support_idx]
            support_mask = self.support_labels[support_idx]

            query_img = self.query_images[pat_idx]
            query_mask = self.query_images[pat_idx]

        new_support_mask = np.zeros_like(support_mask, shape=support_mask.shape)
        new_support_mask[support_mask == label] = 1

        new_query_mask = np.zeros_like(query_mask, shape=query_mask.shape)
        new_query_mask[query_mask == label] = 1

        support_img = (support_img - support_img.mean()) / support_img.std()
        query_img = (query_img - query_img.mean()) / query_img.std()

        support_imgs = [[support_img]]
        support_imgs = np.stack((support_imgs, support_imgs, support_imgs), axis=2)

        query_imgs = [query_img]
        query_imgs = np.stack((query_imgs, query_imgs, query_imgs), axis=1)

        new_support_masks = np.stack([[new_support_mask]], axis=0)
        new_query_masks = np.stack([new_query_mask], axis=0)

        # gamma transform
        if np.random.random(1) > 0.5:
            query_imgs = self.gamma_tansform(query_imgs)
        else:
            support_imgs = self.gamma_tansform(support_imgs)

        # geom transform
        if np.random.random(1) > 0.5:
            query_imgs, new_query_masks = self.geom_transform(
                query_imgs, new_query_masks
            )
        else:
            (
                support_imgs,
                new_support_masks,
            ) = self.geom_transform(support_imgs, new_support_masks)

        sample = {
            "support_images": support_imgs,
            "support_fg_labels": new_support_masks,
            "query_images": query_imgs,
            "query_labels": new_query_masks,
        }

        return sample
