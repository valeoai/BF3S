config = {}
# set the parameters related to the training and testing set

num_classes = 389

data_train_opt = {
    "dataset_name": "ImageNetLowShot",
    "nKnovel": 0,
    "nKbase": num_classes,
    "n_exemplars": 0,
    "n_test_novel": 0,
    "n_test_base": 128,
    "batch_size": 1,
    "epoch_size": 4000,
    "dataset_args": {"phase": "train"},
    "phase": "train",
}

data_test_opt = {
    "dataset_name": "ImageNetLowShot",
    "nKnovel": 200,
    "nKbase": num_classes,
    "n_exemplars": 1,
    "n_test_novel": 200,
    "n_test_base": 0,
    "batch_size": 1,
    "epoch_size": 500,
    "dataset_args": {"phase": "val"},
}
config["data_train_opt"] = data_train_opt
config["data_test_opt"] = data_test_opt


LUT_lr = [(60, 0.1), (90, 0.01), (120, 0.001), (130, 0.0001)]
config["max_num_epochs"] = 130

networks = {}
net_optim_paramsF = {
    "optim_type": "sgd",
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "LUT_lr": LUT_lr,
}
networks["feature_extractor"] = {
    "def_file": "feature_extractors.resnet",
    "pretrained": None,
    "opt": {"arch": "resnet10", "pool": "none"},
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
    "num_classes": 1000,
    "num_features": 512,
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
    "convnet_type": "resnet_block",
    "convnet_opt": {
        "block_type": "BasicBlock",
        "inplanes": 512,
        "planes": 512,
        "num_layers": 2,
        "stride": 1,
    },
    "classifier_opt": {
        "classifier_type": "cosine",
        "num_channels": 512,
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

criterions = {"loss": {"ctype": "CrossEntropyLoss", "opt": None}}
config["criterions"] = criterions

config["algorithm_type"] = "selfsupervision.fewshot_selfsupervision_rotation"
config["auxiliary_rotation_task_coef"] = 1.0
config["rotation_invariant_classifier"] = False
config["random_rotation"] = False
