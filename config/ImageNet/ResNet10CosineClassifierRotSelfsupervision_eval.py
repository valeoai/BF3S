"""
This config file is used only for evaluating the ResN10-based CC+rot model
trained using the config file:
./config/ImageNet/ResNet10CosineClassifierRotSelfsupervision.py
"""

config = {}
# set the parameters related to the training and testing set

nKbase = 389
nKnovel = 200
n_exemplars = 1

data_train_opt = {
    "nKnovel": nKnovel,
    "nKbase": nKbase,
    "n_exemplars": n_exemplars,
    "n_test_novel": nKnovel,
    "n_test_base": nKnovel,
    "batch_size": 4,
    "epoch_size": 4000,
    "data_dir": "./datasets/feature_datasets/ImageNet/ResNet10CosineClassifierRotSelfsupervision",
}

config["data_train_opt"] = data_train_opt
config["max_num_epochs"] = 4

nFeat = 512

networks = {
    "feature_extractor": {
        "def_file": "feature_extractors.dumb_feat",
        "pretrained": None,
        "opt": {"dropout": 0},
        "optim_params": None,
    }
}

net_optim_paramsC = {
    "optim_type": "sgd",
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "LUT_lr": [(2, 0.01), (4, 0.001)],
}
pretrainedC = (
    "./experiments/ImageNet/ResNet10CosineClassifierRotSelfsupervision/classifier_net_epoch130"
)

net_optionsC = {
    "num_classes": 1000,
    "num_features": 512,
    "scale_cls": 10,
    "learn_scale": True,
    "global_pooling": True,
}
networks["classifier"] = {
    "def_file": "classifiers.cosine_classifier_with_weight_generator",
    "pretrained": pretrainedC,
    "opt": net_optionsC,
    "optim_params": net_optim_paramsC,
}
config["networks"] = networks

criterions = {"loss": {"ctype": "CrossEntropyLoss", "opt": None}}
config["criterions"] = criterions

config[
    "data_dir"
] = "./datasets/feature_datasets/ImageNet/ResNet10CosineClassifierRotSelfsupervision"
