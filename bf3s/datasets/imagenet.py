import json
import os
import os.path

import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import ImageEnhance

import bf3s.utils as utils

# Set the appropriate paths of the datasets here.
_IMAGENET_DATASET_DIR = "./datasets/ImageNet"
_IMAGENET256_DATASET_DIR = "/datasets_local/ImageNet256"
_IMAGENET_LOWSHOT_BENCHMARK_CATEGORY_SPLITS_PATH = (
    "./data/IMAGENET_LOWSHOT_BENCHMARK_CATEGORY_SPLITS.json"
)
_MEAN_PIXEL = [0.485, 0.456, 0.406]
_STD_PIXEL = [0.229, 0.224, 0.225]


def load_ImageNet_fewshot_split(class_names):
    with open(_IMAGENET_LOWSHOT_BENCHMARK_CATEGORY_SPLITS_PATH, "r") as f:
        label_idx = json.load(f)

    assert len(label_idx["label_names"]) == len(class_names)

    def get_class_indices(class_indices1):
        class_indices2 = []
        for index in class_indices1:
            class_name_this = label_idx["label_names"][index]
            assert class_name_this in class_names
            class_indices2.append(class_names.index(class_name_this))

        class_names_tmp1 = [label_idx["label_names"][index] for index in class_indices1]
        class_names_tmp2 = [class_names[index] for index in class_indices2]

        assert class_names_tmp1 == class_names_tmp2

        return class_indices2

    base_classes = get_class_indices(label_idx["base_classes"])
    base_classes_val = get_class_indices(label_idx["base_classes_1"])
    base_classes_test = get_class_indices(label_idx["base_classes_2"])
    novel_classes_val = get_class_indices(label_idx["novel_classes_1"])
    novel_classes_test = get_class_indices(label_idx["novel_classes_2"])

    return (
        base_classes,
        base_classes_val,
        base_classes_test,
        novel_classes_val,
        novel_classes_test,
    )


class ImageJitter:
    def __init__(self, transformdict):
        transformtypedict = dict(
            Brightness=ImageEnhance.Brightness,
            Contrast=ImageEnhance.Contrast,
            Sharpness=ImageEnhance.Sharpness,
            Color=ImageEnhance.Color,
        )
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert("RGB")

        return out


class ImageNetBase(data.Dataset):
    def __init__(self, split="train", size256=False, transform=None):

        dataset_name = "ImageNet256" if size256 else "ImageNet"
        assert (split in ("train", "val")) or (split.find("train_subset") != -1)
        self.split = split
        self.name = f"{dataset_name}_Split_" + self.split

        data_dir = _IMAGENET256_DATASET_DIR if size256 else _IMAGENET_DATASET_DIR
        print(f"==> Loading {dataset_name} dataset - split {self.split}")
        print(f"==> {dataset_name} directory: {data_dir}")

        self.transform = transform
        print(f"==> transform: {self.transform}")
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        split_dir = train_dir if (self.split.find("train") != -1) else val_dir
        self.data = datasets.ImageFolder(split_dir, self.transform)
        self.labels = [item[1] for item in self.data.imgs]

        if self.split.find("train_subset") != -1:
            subsetK = int(self.split[len("train_subset") :])
            assert subsetK > 0
            self.split = "train"

            label2ind = utils.build_label_index(self.data.targets)
            all_indices = []
            for label, img_indices in label2ind.items():
                assert len(img_indices) >= subsetK
                all_indices += img_indices[:subsetK]

            self.data.imgs = [self.data.imgs[idx] for idx in all_indices]
            self.data.samples = [self.data.samples[idx] for idx in all_indices]
            self.data.targets = [self.data.targets[idx] for idx in all_indices]
            self.labels = [self.labels[idx] for idx in all_indices]

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, label

    def __len__(self):
        return len(self.data)


class ImageNet(ImageNetBase):
    def __init__(
        self,
        split="train",
        use_geometric_aug=True,
        use_simple_geometric_aug=False,
        use_color_aug=True,
        cutout_length=0,
        do_not_use_random_transf=False,
        size256=False,
    ):

        transform_train = []
        assert not (use_simple_geometric_aug and use_geometric_aug)
        if use_geometric_aug:
            transform_train.append(transforms.RandomResizedCrop(224))
            transform_train.append(transforms.RandomHorizontalFlip())
        elif use_simple_geometric_aug:
            transform_train.append(transforms.Resize(256))
            transform_train.append(transforms.RandomCrop(224))
            transform_train.append(transforms.RandomHorizontalFlip())
        else:
            transform_train.append(transforms.Resize(256))
            transform_train.append(transforms.CenterCrop(224))

        if use_color_aug:
            jitter_params = {"Brightness": 0.4, "Contrast": 0.4, "Color": 0.4}
            transform_train.append(ImageJitter(jitter_params))

        transform_train.append(lambda x: np.asarray(x))
        transform_train.append(transforms.ToTensor())
        transform_train.append(transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL))

        if cutout_length > 0:
            print(f"==> cutout_length: {cutout_length}")
            transform_train.append(utils.Cutout(n_holes=1, length=cutout_length))

        transform_train = transforms.Compose(transform_train)
        self.transform_train = transform_train

        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL),
            ]
        )

        if do_not_use_random_transf or split == "val":
            transform = transform_test
        else:
            transform = transform_train

        super().__init__(split=split, size256=size256, transform=transform)


class ImageNetLowShot(ImageNet):
    def __init__(self, phase="train", split="train", do_not_use_random_transf=False):

        assert phase in ("train", "test", "val")
        assert split in ("train", "val")

        use_aug = (phase == "train") and (do_not_use_random_transf == False)

        super().__init__(split=split, use_geometric_aug=use_aug, use_color_aug=use_aug)

        self.phase = phase
        self.split = split
        self.name = "ImageNetLowShot_Phase_" + phase + "_Split_" + split
        print(f"==> Loading ImageNet few-shot benchmark - phase {phase}")

        # ***********************************************************************
        (
            base_classes,
            _,
            _,
            novel_classes_val,
            novel_classes_test,
        ) = load_ImageNet_fewshot_split(self.data.classes)
        # ***********************************************************************

        self.label2ind = utils.build_label_index(self.labels)
        self.labelIds = sorted(self.label2ind.keys())
        self.num_cats = len(self.labelIds)
        assert self.num_cats == 1000

        self.labelIds_base = base_classes
        self.num_cats_base = len(self.labelIds_base)
        if self.phase == "val" or self.phase == "test":
            self.labelIds_novel = (
                novel_classes_val if (self.phase == "val") else novel_classes_test
            )
            self.num_cats_novel = len(self.labelIds_novel)

            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert len(intersection) == 0


class ImageNetLowShotFeatures:
    def __init__(self, data_dir, image_split="train", phase="train"):
        # data_dir: path to the directory with the saved ImageNet features.
        # image_split: the image split of the ImageNet that will be loaded.
        # phase: whether the dataset will be used for training, validating, or
        # testing the few-shot model model.
        assert image_split in ("train", "val")
        assert phase in ("train", "val", "test")

        self.phase = phase
        self.image_split = image_split
        self.name = (
            f"ImageNetLowShotFeatures_ImageSplit_{self.image_split}" f"_Phase_{self.phase}"
        )

        dataset_file = os.path.join(data_dir, "ImageNet_" + self.image_split + ".h5")
        self.data_file = h5py.File(dataset_file, "r")
        self.count = self.data_file["count"][0]
        self.features = self.data_file["all_features"][...]
        self.labels = self.data_file["all_labels"][: self.count].tolist()

        # ***********************************************************************
        data_tmp = datasets.ImageFolder(os.path.join(_IMAGENET_DATASET_DIR, "train"), None)
        (
            base_classes,
            base_classes_val,
            base_classes_test,
            novel_classes_val,
            novel_classes_test,
        ) = load_ImageNet_fewshot_split(data_tmp.classes)
        # ***********************************************************************

        self.label2ind = utils.build_label_index(self.labels)
        self.labelIds = sorted(self.label2ind.keys())
        self.num_cats = len(self.labelIds)
        assert self.num_cats == 1000

        self.labelIds_base = base_classes
        self.num_cats_base = len(self.labelIds_base)

        if self.phase == "val" or self.phase == "test":
            self.labelIds_novel = (
                novel_classes_val if (self.phase == "val") else novel_classes_test
            )
            self.num_cats_novel = len(self.labelIds_novel)

            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert len(intersection) == 0
            self.base_classes_eval_split = (
                base_classes_val if (self.phase == "val") else base_classes_test
            )

    def __getitem__(self, index):
        features_this = torch.Tensor(self.features[index]).view(-1, 1, 1)
        label_this = self.labels[index]
        return features_this, label_this

    def __len__(self):
        return int(self.count)
