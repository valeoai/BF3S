"""Tests a few-shot models on the Imagenet-based few-shot dataset[1,2].

Example of usage:
# Test the CC+Rot model on the 1-shot setting.
$ python scripts/test_fewshot_imagenet.py --config=ImageNet/ResNet10CosineClassifierRotSelfsupervision_eval --testset --num_train=1 --bias_novel=0.8
# ==> Top 5 Accuracies:      [Novel: 46.43 | Base: 93.52 | All 57.88 | Novel vs All 42.39 | Base vs All 82.45 | All prior 57.42]

# Test the CC+Rot model on the 2-shot setting.
$ python scripts/test_fewshot_imagenet.py --config=ImageNet/ResNet10CosineClassifierRotSelfsupervision_eval --testset --num_train=2 --bias_novel=0.75
# ==> Top 5 Accuracies:      [Novel: 57.80 | Base: 93.52 | All 64.76 | Novel vs All 53.88 | Base vs All 82.02 | All prior 64.11]

# Test the CC+Rot model on the 5-shot setting.
$ python scripts/test_fewshot_imagenet.py --config=ImageNet/ResNet10CosineClassifierRotSelfsupervision_eval --testset --num_train=5 --bias_novel=0.7
# ==> Top 5 Accuracies:      [Novel: 69.67 | Base: 93.52 | All 72.29 | Novel vs All 66.43 | Base vs All 81.59 | All prior 71.53]

# Test the CC+Rot model on the 10-shot setting.
$ python scripts/test_fewshot_imagenet.py --config=ImageNet/ResNet10CosineClassifierRotSelfsupervision_eval --testset --num_train=10 --bias_novel=0.65
# ==> Top 5 Accuracies:      [Novel: 74.64 | Base: 93.52 | All 75.63 | Novel vs All 71.03 | Base vs All 82.94 | All prior 74.77]

# Test the CC+Rot model on the 20-shot setting.
$ python scripts/test_fewshot_imagenet.py --config=ImageNet/ResNet10CosineClassifierRotSelfsupervision_eval --testset --num_train=20 --bias_novel=0.6
# ==> Top 5 Accuracies:      [Novel: 77.31 | Base: 93.52 | All 77.40 | Novel vs All 72.67 | Base vs All 84.91 | All prior 76.46]

The config argument specifies the model that will be evaluated.

[1] B. Hariharan and R. Girshick. Low-shot visual recognition by shrinking and hallucinating features.
[2] Y.-X. Wang and R. Girshick, M. Hebert, B. Hariharan. Low-shot learning from imaginary data.
"""


import argparse
import os

from bf3s import project_root
from bf3s.algorithms.fewshot.imagenet_lowshot import ImageNetLowShot
from bf3s.dataloaders.dataloader_fewshot import LowShotDataloader
from bf3s.datasets.imagenet import ImageNetLowShotFeatures


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
        help="config file with parameters of the experiment",
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
        "--testset",
        default=False,
        action="store_true",
        help="If True, the model is evaluated on the test set of "
        "ImageNetLowShot. Otherwise, the validation set is used for "
        "evaluation.",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=100, help="the number of test episodes."
    )
    parser.add_argument("--prior", type=float, default=0.7)
    parser.add_argument(
        "--num_train",
        type=int,
        default=1,
        help="number of training examples per novel class.",
    )
    parser.add_argument(
        "--bias_novel",
        default=1.0,
        type=float,
        help="bias for the classification scores of the novel classes " "(multiplicative).",
    )
    parser.add_argument("--batch_size", type=int, default=1000)

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

    if args_opt.bias_novel != 1.0:
        config["networks"]["classifier"]["opt"]["bias_novel"] = args_opt.bias_novel

    algorithm = ImageNetLowShot(config)
    if args_opt.cuda:  # enable cuda.
        algorithm.load_to_gpu()

    if args_opt.checkpoint != 0:  # load checkpoint.
        algorithm.load_checkpoint(
            epoch=args_opt.checkpoint if (args_opt.checkpoint > 0) else "*",
            train=False,
            suffix="",
        )

    # Prepare the datasets and the the dataloader.
    n_exemplars = data_train_opt = config["data_train_opt"]["n_exemplars"]
    if args_opt.num_train > 0:
        n_exemplars = args_opt.num_train

    eval_phase = "test" if args_opt.testset else "val"
    data_train_opt = config["data_train_opt"]
    feat_data_train = ImageNetLowShotFeatures(
        data_dir=data_train_opt["data_dir"], image_split="train", phase=eval_phase
    )
    feat_data_test = ImageNetLowShotFeatures(
        data_dir=data_train_opt["data_dir"], image_split="val", phase=eval_phase
    )
    data_loader = LowShotDataloader(
        feat_data_train,
        feat_data_test,
        n_exemplars=args_opt.num_train,
        batch_size=args_opt.batch_size,
        num_workers=0,
    )

    results = algorithm.evaluate(
        data_loader, num_eval_exp=args_opt.num_episodes, prior=args_opt.prior, suffix="best"
    )

    algorithm.logger.info("==> algorithm_type: {}".format("ImageNetLowShot"))
    algorithm.logger.info(f"==> num_train: {args_opt.num_train}")
    algorithm.logger.info(f"==> num_episodes: {args_opt.num_episodes}")
    algorithm.logger.info(f"==> eval_phase: {eval_phase}")
    algorithm.logger.info(f"==> bias_novel: {args_opt.bias_novel}")
    algorithm.logger.info(f"==> results: {results}")


if __name__ == "__main__":
    main()
