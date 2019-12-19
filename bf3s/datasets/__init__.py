from bf3s.datasets.cifar100_fewshot import CIFAR100FewShot
from bf3s.datasets.imagenet import ImageNet
from bf3s.datasets.imagenet import ImageNetLowShot
from bf3s.datasets.mini_imagenet import MiniImageNet
from bf3s.datasets.mini_imagenet import MiniImageNet3x3Patches
from bf3s.datasets.mini_imagenet import MiniImageNet80x80
from bf3s.datasets.mini_imagenet import MiniImageNetImagesAnd3x3Patches
from bf3s.datasets.tiered_mini_imagenet import tieredMiniImageNet
from bf3s.datasets.tiered_mini_imagenet import tieredMiniImageNet80x80


all_datasets = dict(
    CIFAR100FewShot=CIFAR100FewShot,
    MiniImageNet=MiniImageNet,
    MiniImageNet80x80=MiniImageNet80x80,
    MiniImageNet3x3Patches=MiniImageNet3x3Patches,
    MiniImageNetImagesAnd3x3Patches=MiniImageNetImagesAnd3x3Patches,
    tieredMiniImageNet=tieredMiniImageNet,
    tieredMiniImageNet80x80=tieredMiniImageNet80x80,
    ImageNet=ImageNet,
    ImageNetLowShot=ImageNetLowShot,
)


def dataset_factory(dataset_name, *args, **kwargs):
    return all_datasets[dataset_name](*args, **kwargs)
