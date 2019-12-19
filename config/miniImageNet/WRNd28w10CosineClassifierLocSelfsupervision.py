config = {}
# set the parameters related to the training and testing set

nKbase = 64


data_train_opt = {
    "dataset_name": "MiniImageNet80x80",
    "nKnovel": 0,
    "nKbase": nKbase,
    "n_exemplars": 0,
    "n_test_novel": 0,
    "n_test_base": 64,
    "batch_size": 1,
    "epoch_size": 1000,
    "phase": "train",
}

data_train_patches_opt = {
    "dataset_name": "MiniImageNet3x3Patches",
    "batch_size_unlabeled": 64,
    "epoch_size_unlabeled": 64_000,
    "dataset_args": {"image_size": 96, "patch_jitter": 8, "phase": "train",},
    "parallel_datasets": False,
}

data_test_opt = {
    "dataset_name": "MiniImageNet80x80",
    "nKnovel": 5,
    "nKbase": nKbase,
    "n_exemplars": 1,
    "n_test_novel": 15 * 5,
    "n_test_base": 15 * 5,
    "batch_size": 1,
    "epoch_size": 500,
}

config["benchmark"] = "MiniImageNet80x80"
config["data_train_opt"] = data_train_opt
config["data_test_opt"] = data_test_opt
config["data_train_patches_opt"] = data_train_patches_opt


config["max_num_epochs"] = 26
LUT_lr = [(20, 0.1), (23, 0.01), (26, 0.001)]

networks = {}
net_optionsF = {"depth": 28, "widen_Factor": 10, "drop_rate": 0.0, "pool": "avg"}
net_optim_paramsF = {
    "optim_type": "sgd",
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "LUT_lr": LUT_lr,
}
networks["feature_extractor"] = {
    "def_file": "feature_extractors.wide_resnet",
    "pretrained": None,
    "opt": net_optionsF,
    "optim_params": net_optim_paramsF,
}

net_optim_paramsC = {
    "optim_type": "sgd",
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "LUT_lr": LUT_lr,
}
net_optionsC = {
    "num_classes": nKbase,
    "num_features": 640,
    "scale_cls": 10,
    "learn_scale": True,
    "global_pooling": False,
}
networks["classifier"] = {
    "def_file": "classifiers.cosine_classifier_with_weight_generator",
    "pretrained": None,
    "opt": net_optionsC,
    "optim_params": net_optim_paramsC,
}

net_optim_paramsC = {
    "optim_type": "sgd",
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "LUT_lr": LUT_lr,
}
net_optionsC = {
    "classifier_type": "mlp_linear",
    "num_classes": 8,
    "num_channels": 640 * 2,
    "scale_cls": 1,
    "mlp_channels": [1280,],
}
networks["classifier_loc"] = {
    "def_file": "classifiers.classifier",
    "pretrained": None,
    "opt": net_optionsC,
    "optim_params": net_optim_paramsC,
}

net_optim_paramsC = {
    "optim_type": "sgd",
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "LUT_lr": LUT_lr,
}
net_optionsC = {
    "classifier_type": "cosine",
    "num_classes": nKbase,
    "num_channels": 640 * 1 * 1,
    "scale_cls": 10,
    "learn_scale": True,
    "global_pooling": True,
}
networks["patch_classifier"] = {
    "def_file": "classifiers.classifier",
    "pretrained": None,
    "opt": net_optionsC,
    "optim_params": net_optim_paramsC,
}

config["networks"] = networks

criterions = {}
criterions["loss"] = {"ctype": "CrossEntropyLoss", "opt": None}
config["criterions"] = criterions
config["algorithm_type"] = "selfsupervision.fewshot_selfsupervision_patch_location"

config["patch_classification_loss_coef"] = 1.0
config["patch_location_loss_coef"] = 1.0
config["standardize_image"] = True
config["standardize_patches"] = True
config["combine_patches"] = "average"
