config = {}
# set the parameters related to the training and testing set

num_classes = 64

data_train_opt = {
    "dataset_name": "CIFAR100FewShot",
    "nKnovel": 0,
    "nKbase": num_classes,
    "n_exemplars": 0,
    "n_test_novel": 0,
    "n_test_base": 64,
    "batch_size": 1,
}
data_train_opt["epoch_size"] = data_train_opt["batch_size"] * 1000
data_train_opt["phase"] = "train"

data_test_opt = {}
data_test_opt["dataset_name"] = "CIFAR100FewShot"
data_test_opt["nKnovel"] = 5
data_test_opt["nKbase"] = num_classes
data_test_opt["n_exemplars"] = 1
data_test_opt["n_test_novel"] = 15 * data_test_opt["nKnovel"]
data_test_opt["n_test_base"] = 15 * data_test_opt["nKnovel"]
data_test_opt["batch_size"] = 1
data_test_opt["epoch_size"] = 1000

config["data_train_opt"] = data_train_opt
config["data_test_opt"] = data_test_opt

LUT_lr = [(20, 0.1), (40, 0.01), (60, 0.001)]
# In almost all cases the best model in the validation set was the one after the
# first epoch with learning rate 0.01, i.e., the model of the 21st epoch. So, to
# avoid unnecessarily training for more epochs, we stop at the 21 epoch.
config["max_num_epochs"] = 21

networks = {}
net_optionsF = {
    "depth": 28,
    "widen_Factor": 10,
    "drop_rate": 0.0,
    "pool": "none",
    "strides": [1, 2, 2],
}
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
    "num_classes": 100,
    "num_features": 640,
    "scale_cls": 10,
    "learn_scale": True,
    "global_pooling": True,
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
    "convnet_type": "wrn_block",
    "convnet_opt": {
        "num_channels_in": 640,
        "num_channels_out": 640,
        "num_layers": 4,
        "stride": 2,
    },
    "classifier_opt": {
        "classifier_type": "cosine",
        "num_channels": 640,
        "scale_cls": 10.0,
        "learn_scale": True,
        "num_classes": 4,
        "global_pooling": True,
    },
}
networks["classifier_aux"] = {
    "def_file": "classifiers.convnet_plus_classifier",
    "pretrained": None,
    "opt": net_optionsC,
    "optim_params": net_optim_paramsC,
}

config["networks"] = networks

criterions = {}
criterions["loss"] = {"ctype": "CrossEntropyLoss", "opt": None}
config["criterions"] = criterions

config["algorithm_type"] = "selfsupervision.fewshot_selfsupervision_rotation"
config["auxiliary_rotation_task_coef"] = 1.0
config["rotation_invariant_classifier"] = True
config["random_rotation"] = False
