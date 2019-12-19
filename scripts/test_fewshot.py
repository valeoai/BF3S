"""Tests a few-shot recognition models.

Example of usage:
# Test the CC+Rot model on MiniImageNet and the 5-way 1-shot setting
$ python scripts/test_fewshot.py --config=miniImageNet/WRNd28w10CosineClassifierRotAugRotSelfsupervision --num_novel=5 --num_train=1 --num_episodes=2000
# Test the CC+Rot model on MiniImageNet and the 5-way 5-shot setting
$ python scripts/test_fewshot.py --config=miniImageNet/WRNd28w10CosineClassifierRotAugRotSelfsupervision --num_novel=5 --num_train=5 --num_episodes=2000

The config argument specifies the model that will be tested.
"""


import argparse
import os

from bf3s import project_root
from bf3s.algorithms.selfsupervision.fewshot_selfsupervision_patch_location import (
    FewShotPatchLocationSelfSupervision,
)
from bf3s.algorithms.selfsupervision.fewshot_selfsupervision_rotation import (
    FewShotRotationSelfSupervision,
)
from bf3s.dataloaders.dataloader_fewshot import FewShotDataloader
from bf3s.datasets import all_datasets


def get_arguments():
    """ Parse input arguments. """
    parser = argparse.ArgumentParser(
        description="Code for testing few-shot image recogntion models."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        default="",
        help="config file with parameters of the experiment.",
    )
    parser.add_argument("--cuda", type=bool, default=True, help="enables cuda.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="the split that will be used for the testing the model.",
    )
    parser.add_argument("--num_novel", type=int, default=5, help="number of novel classes.")
    parser.add_argument(
        "--num_train",
        type=int,
        default=1,
        help="number of training examples per novel class.",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=2000, help="number of test episodes."
    )
    parser.add_argument(
        "--nocheckpoint", default=False, action="store_true", help="uses an untrained model.",
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

    # Set train and test datasets and the corresponding data loaders
    data_test_opt = config["data_test_opt"]
    assert args_opt.split in ("test", "val")
    dataset_test_args = {"phase": args_opt.split}
    dataset_test = all_datasets[data_test_opt["dataset_name"]](**dataset_test_args)

    # Number of test examples from all novel classes.
    num_test = 15 * args_opt.num_novel
    # Number of test examples from all base classes.
    num_test_base = data_test_opt["n_test_base"]
    num_base = data_test_opt["nKbase"]
    dloader_test = FewShotDataloader(
        dataset=dataset_test,
        nKnovel=args_opt.num_novel,
        nKbase=num_base,
        n_exemplars=args_opt.num_train,
        n_test_novel=num_test,
        n_test_base=num_test_base,
        batch_size=1,
        num_workers=0,
        epoch_size=args_opt.num_episodes,
    )

    algorithm_type = (
        config["algorithm_type"]
        if ("algorithm_type" in config)
        else "selfsupervision.fewshot_selfsupervision_rotation"
    )

    if algorithm_type == "selfsupervision.fewshot_selfsupervision_patch_location":
        algorithm = FewShotPatchLocationSelfSupervision(config)
    else:
        algorithm = FewShotRotationSelfSupervision(config)

    if args_opt.cuda:  # enable cuda
        algorithm.load_to_gpu()

    # Evaluate the checkpoint with the highest few-shot classification
    # accuracy on the validation set.
    if not args_opt.nocheckpoint:  # load checkpoint
        algorithm.load_checkpoint(epoch="*", train=False, suffix=".best")

    # Run evaluation.
    algorithm.logger.info(f"==> algorithm_type: {algorithm_type}")
    algorithm.logger.info(f"==> num_novel: {args_opt.num_novel}")
    algorithm.logger.info(f"==> num_train: {args_opt.num_train}")
    algorithm.logger.info(f"==> num_test: {num_test}")
    algorithm.logger.info(f"==> num_test_base: {num_test_base}")
    algorithm.logger.info(f"==> num_episodes: {args_opt.num_episodes}")
    algorithm.logger.info(f"==> split: {args_opt.split}")
    algorithm.evaluate(dloader_test)


if __name__ == "__main__":
    main()
