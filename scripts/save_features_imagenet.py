"""Extracts and saves features from the images of the ImageNet dataset.

Example of usage:
# Extract features from the validation image split of Imagenet.
$ python scripts/save_features_imagenet.py --config=ImageNet/ResNet10CosineClassifierRotSelfsupervision --split='train'
# Extract features from the training image split of Imagenet.
$ python scripts/save_features_imagenet.py --config=ImageNet/ResNet10CosineClassifierRotSelfsupervision --split='val'

The config argument specifies the model that will be used.
"""


import argparse
import os

from bf3s import project_root
from bf3s.algorithms.utils.save_features import SaveFeatures
from bf3s.dataloaders.basic_dataloaders import SimpleDataloader
from bf3s.datasets.imagenet import ImageNet


def get_arguments():
    """ Parse input arguments. """
    parser = argparse.ArgumentParser(
        description="Code for extracting and saving features from ImageNet."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        default="",
        help="config file with hyper-parameters of the model that we will use "
        "for extracting features from ImageNet dataset.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=0,
        help="checkpoint (epoch id) that will be loaded. If a negative value "
        "is given then the latest existing checkpoint is loaded.",
    )
    parser.add_argument("--cuda", type=bool, default=True, help="enables cuda.")
    parser.add_argument(
        "--num_workers", type=int, default=8, help="number of workers for the data loader(s).",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="the ImageNet split from which the features will extracted and " "saved.",
    )

    args_opt = parser.parse_args()
    exp_base_directory = os.path.join(project_root, "experiments")
    exp_directory = os.path.join(exp_base_directory, args_opt.config)

    # Load the configuration params of the experiment
    exp_config_file = "config." + args_opt.config.replace("/", ".")
    print(f"Loading experiment {args_opt.config}")
    config = __import__(exp_config_file, fromlist=[""]).config
    config["exp_dir"] = exp_directory  # where logs, models, etc will be stored.
    print(f"Generated logs and/or snapshots will be stored on {exp_directory}")

    return args_opt, exp_directory, config


def main():
    args_opt, exp_directory, config = get_arguments()

    if (args_opt.split != "train") and (args_opt.split != "val"):
        raise ValueError(f"Not valid split {args_opt.split}")

    dataset = ImageNet(split=args_opt.split, use_geometric_aug=False, use_color_aug=False)
    dloader = SimpleDataloader(
        dataset, batch_size=args_opt.batch_size, train=False, num_workers=args_opt.num_workers,
    )

    algorithm = SaveFeatures(config)
    if args_opt.cuda:  # enable cuda
        algorithm.load_to_gpu()

    # Load (the latest) checkpoint.
    algorithm.load_checkpoint(
        epoch=(args_opt.checkpoint if (args_opt.checkpoint > 0) else "*"), train=False
    )

    dst_directory = os.path.join(project_root, "datasets", "feature_datasets", args_opt.config)

    algorithm.logger.info(f"==> Destination directory: {dst_directory}")
    if not os.path.isdir(dst_directory):
        os.makedirs(dst_directory)

    dst_filename = os.path.join(dst_directory, "ImageNet_" + args_opt.split + ".h5")

    algorithm.logger.info(f"==> dst_filename: {dst_filename}")

    algorithm.save_features(
        dataloader=dloader, filename=dst_filename, feature_name=None, global_pooling=True
    )


if __name__ == "__main__":
    main()
