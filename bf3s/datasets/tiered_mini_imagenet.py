import csv
import os
import os.path

import cv2
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

import bf3s.utils as utils

# Set the appropriate paths of the datasets here.
_TIERED_MINI_IMAGENET_DATASET_DIR = "./datasets/tieredMiniImageNet"
_TIERED_MIN_IMAGENET_METADATA_DIR = "./data/tiered_imagenet_split"
_MIN_IMAGENET_METADATA_DIR = "./data/mini_imagenet_split"


def read_csv_class_file(filename):
    data = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            data.append(row)

    return data


def get_train_class_names_of_tiered_mini_imagenet():
    file_train_class_meta_data = os.path.join(_TIERED_MIN_IMAGENET_METADATA_DIR, "train.csv")
    meta_data_train = read_csv_class_file(file_train_class_meta_data)
    train_class_names = [class_name for (class_name, _) in meta_data_train]
    return train_class_names


def get_all_class_names_of_mini_imagenet():
    file_train_class_meta_data = os.path.join(
        _MIN_IMAGENET_METADATA_DIR, "train_val_test_classnames.csv"
    )
    meta_data_all = read_csv_class_file(file_train_class_meta_data)
    all_class_names = [class_name for (class_name,) in meta_data_all]
    return all_class_names


def tiered_train_classes_not_miniimagenet():
    tiered_classes = get_train_class_names_of_tiered_mini_imagenet()
    mini_classes = get_all_class_names_of_mini_imagenet()
    print(f"==> tiered-MiniImageNet train classes: {tiered_classes}")
    print(f"==> MiniImageNet all classes: {mini_classes}")

    tiered_classes_not_mini_imagenet = [True] * len(tiered_classes)
    counter = 0
    for i, t, in enumerate(tiered_classes):
        if t in mini_classes:
            tiered_classes_not_mini_imagenet[i] = False
            counter += 1

    print(f"==> {counter} classes removed from tiered-MiniImageNet train classes.")

    return tiered_classes_not_mini_imagenet


class tieredMiniImageNet(data.Dataset):
    def __init__(
        self, phase="train", load_single_file_split=False, do_not_use_random_transf=False
    ):

        data_dir = _TIERED_MINI_IMAGENET_DATASET_DIR

        class_splits = ["train", "val", "test"]
        files_images = {}
        files_labels = {}
        for class_split in class_splits:
            files_images[class_split] = os.path.join(data_dir, f"{class_split}_images_png.pkl")
            files_labels[class_split] = os.path.join(data_dir, f"{class_split}_labels.pkl")

        self.phase = phase
        if load_single_file_split:
            assert self.phase in ("train", "val", "test")

            self.name = "tieredMiniImagenet_" + self.phase
            print(f"Loading tiered Mini ImageNet {self.phase}")

            images = utils.load_pickle_data(files_images[self.phase])
            labels = utils.load_pickle_data(files_labels[self.phase])
            labels = labels["label_specific"].tolist()

            self.data = images
            self.labels = labels
            self.label2ind = utils.build_label_index(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)

        else:
            assert phase in ("train", "trainval", "train_not_miniimagnet", "val", "test")

            self.name = "tieredMiniImagenet_" + phase
            print(f"Loading tiered Mini ImageNet {phase}")

            labels_train = utils.load_pickle_data(files_labels["train"])
            num_train_classes = labels_train["label_specific"].max() + 1

            labels_val = utils.load_pickle_data(files_labels["val"])
            num_val_classes = labels_val["label_specific"].max() + 1

            labels_test = utils.load_pickle_data(files_labels["test"])
            num_test_classes = labels_test["label_specific"].max() + 1
            print("==> tiered MiniImageNet:")
            print(f"====> number of train classes : {num_train_classes}")
            print(f"====> number of validation classes : {num_val_classes}")
            print(f"====> number of test classes : {num_test_classes}")
            if self.phase == "train":
                # During training phase we only load the training phase images
                # of the training categories (aka base categories).
                images_train = utils.load_pickle_data(files_images["train"])
                labels_train_list = labels_train["label_specific"].tolist()
                self.data = images_train
                self.labels = labels_train_list

                self.label2ind = utils.build_label_index(self.labels)
                self.labelIds = sorted(self.label2ind.keys())
                self.num_cats = len(self.labelIds)
                self.labelIds_base = self.labelIds
                self.num_cats_base = len(self.labelIds_base)
            elif self.phase == "train_not_miniimagnet":
                images_train = utils.load_pickle_data(files_images["train"])
                labels_train_list = labels_train["label_specific"].tolist()
                # Filter out the train classes that are in the train, test, or
                # validation set of MiniImageNet.
                label2ind = utils.build_label_index(labels_train_list)
                valid_tiered_classes = tiered_train_classes_not_miniimagenet()

                keep_indices = []
                for label, img_indices in label2ind.items():
                    if valid_tiered_classes[label]:
                        keep_indices += img_indices

                keep_labels_train_list = [labels_train_list[i] for i in keep_indices]
                keep_images_train = [images_train[i] for i in keep_indices]

                self.data = keep_images_train
                self.labels = keep_labels_train_list
                self.label2ind = utils.build_label_index(self.labels)
                self.labelIds = sorted(self.label2ind.keys())
                self.num_cats = len(self.labelIds)
                self.labelIds_base = self.labelIds
                self.num_cats_base = len(self.labelIds_base)
            elif self.phase == "trainval":
                # During training phase we only load the training phase images
                # of the training categories (aka base categories).
                images_train = utils.load_pickle_data(files_images["train"])
                labels_train_list = labels_train["label_specific"].tolist()

                images_val = utils.load_pickle_data(files_images["val"])
                labels_val_list = (labels_val["label_specific"] + num_train_classes).tolist()

                self.data = images_train + images_val
                self.labels = labels_train_list + labels_val_list

                self.label2ind = utils.build_label_index(self.labels)
                self.labelIds = sorted(self.label2ind.keys())
                self.num_cats = len(self.labelIds)
                self.labelIds_base = self.labelIds
                self.num_cats_base = len(self.labelIds_base)

            elif self.phase == "val" or self.phase == "test":

                class_split = "val" if self.phase == "val" else "test"
                images_eval = utils.load_pickle_data(files_images[class_split])
                labels_eval = utils.load_pickle_data(files_labels[class_split])
                labels_eval_list = (labels_eval["label_specific"] + num_train_classes).tolist()

                self.data = images_eval
                self.labels = labels_eval_list

                self.label2ind = utils.build_label_index(self.labels)
                for label in range(num_train_classes):
                    self.label2ind[label] = []

                self.labelIds = sorted(self.label2ind.keys())
                self.num_cats = len(self.labelIds)
                self.labelIds_base = {label for label in range(num_train_classes)}
                self.labelIds_novel = utils.build_label_index(labels_eval_list).keys()
                self.num_cats_base = len(self.labelIds_base)
                self.num_cats_novel = len(self.labelIds_novel)
                intersection = set(self.labelIds_base) & set(self.labelIds_novel)
                assert len(intersection) == 0
            else:
                raise ValueError(f"Not valid phase {self.phase}")

        print("==> {} images were loaded.".format(len(self.labels)))

        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        if (self.phase == "test" or self.phase == "val") or (do_not_use_random_transf == True):
            self.transform = transforms.Compose(
                [lambda x: np.asarray(x), transforms.ToTensor(), normalize]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(84, padding=8),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        img = cv2.cvtColor(cv2.imdecode(img, 1), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


class tieredMiniImageNet80x80(tieredMiniImageNet):
    def __init__(
        self, phase="train", load_single_file_split=False, do_not_use_random_transf=False,
    ):
        super().__init__(
            phase=phase,
            load_single_file_split=load_single_file_split,
            do_not_use_random_transf=do_not_use_random_transf,
        )

        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        if (self.phase == "test" or self.phase == "val") or (do_not_use_random_transf == True):
            self.transform = transforms.Compose(
                [
                    transforms.Resize(80),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(84, padding=8),
                    transforms.Resize(80),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
