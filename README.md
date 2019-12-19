# *Boosting Few-Shot Visual Learning with Self-Supervision*

The current project page provides [pytorch](http://pytorch.org/) code that implements the following ICCV 2019 paper:   
**Title:**      "Boosting Few-Shot Visual Learning with Self-Supervision"    
**Authors:**     Spyros Gidaris, Andrei Bursuc, Nikos Komodakis, Patrick Pérez, and Matthieu Cord        

**Abstract:**  
Few-shot learning and self-supervised learning address different facets of the same problem: how to train a model with little or no labeled data. Few-shot learning aims for optimization methods and models that can learn efficiently to recognize patterns in the low data regime. Self-supervised learning focuses instead on unlabeled data and looks into it for the supervisory signal to feed high capacity deep neural networks. In this work we exploit the complementarity of these two domains and propose an approach for improving few-shot learning through self-supervision. We use self-supervision as an auxiliary task in a few-shot learning pipeline, enabling feature extractors to learn richer and more transferable visual representations while still using few annotated samples. Through self-supervision, our approach can be naturally extended towards using diverse unlabeled data from other datasets in the few-shot setting. We report consistent improvements across an array of architectures, datasets and self-supervision techniques.

If you find the code useful in your research, please consider citing our paper:

```BibTeX
@inproceedings{gidaris2019boosting,
  title={Boosting Few-Shot Visual Learning with Self-Supervision},
  author={Gidaris, Spyros and Bursuc, Andrei and Komodakis, Nikos and P{\'e}rez, Patrick and Cord, Matthieu},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
}
```

### License
This code is released under the MIT License (refer to the LICENSE file for details).

## Preparation

### Pre-requisites
* Python 3.7
* Pytorch >= 1.0.0
* CUDA 10.0 or higher

### Installation

**(1)** Clone the repo:
```bash
$ git clone https://github.com/valeoai/BF3S
```

**(2)** Install this repository and the dependencies using pip:
```bash
$ pip install -e ./BF3S
```

With this, you can edit the BF3S code on the fly and import function
and classes of BF3S in other projects as well.

**(3)** Optional. To uninstall this package, run:
```bash
$ pip uninstall BF3S
```

**(4)** Create *dataset* and *experiment* directories:
```bash
$ cd BF3S
$ mkdir ./datasets
$ mkdir ./experiments
```


You can take a look at the [Dockerfile](./Dockerfile) if you are uncertain
the about steps to install this project.


### Necessary datasets.

**(1) MiniImagenet** To download the MiniImagenet dataset go to this
[github page](https://github.com/gidariss/FewShotWithoutForgetting)
and follow the instructions there. Then, set in
[bf3s/datasets/mini_imagenet_dataset.py](bf3s/datasets/mini_imagenet.py#L16) the path to where the dataset resides in your machine.

**(2) tiered-MiniImagenet** To download the tiered-MiniImagenet dataset go to this
[github page](https://github.com/renmengye/few-shot-ssl-public)
and follow the instructions there. Then, set in
[bf3s/datasets/tiered_mini_imagenet_dataset.py](bf3s/datasets/tiered_mini_imagenet.py#L18) the path to where the dataset resides in your machine.

**(3) ImageNet-FS** Download the ImageNet dataset and set in
[bf3s/datasets/imagenet_dataset.py](bf3s/datasets/imagenet.py#L19) the path to where the dataset resides in your machine.

**(4) CIFAR-FS** The dataset will be automatically downloaded when you run the code.
Set in
[bf3s/datasets/cifar100_fewshot_dataset.py](bf3s/datasets/cifar100_fewshot.py#L16)
the path to where the dataset should be downloaded.

### Download pre-trained models (optional).

**(1)** Download the models trained on the MiniImageNet dataset.

```bash
# Run from the BF3S directory
$ mkdir ./experiments/miniImageNet
$ cd ./experiments/miniImageNet

# WRN-28-10-based Cosine Classifier (CC) with rotation prediction self-supervision model.
$ wget https://github.com/valeoai/BF3S/releases/download/0.1.0/WRNd28w10CosineClassifierRotAugRotSelfsupervision.zip
$ unzip WRNd28w10CosineClassifierRotAugRotSelfsupervision.zip

# WRN-28-10-based CC with rotation prediction self-supervision model trained with extra unlabeled images from tiered-MiniImageNet.
$ wget https://github.com/valeoai/BF3S/releases/download/0.1.0/WRNd28w10CosineClassifierRotAugRotSelfsupervision_SemisupervisedTieredUnlabeled.zip
$ unzip WRNd28w10CosineClassifierRotAugRotSelfsupervision_SemisupervisedTieredUnlabeled.zip

# WRN-28-10-based CC with location prediction self-supervision model.
$ wget https://github.com/valeoai/BF3S/releases/download/0.1.0/WRNd28w10CosineClassifierLocSelfsupervision.zip
$ unzip WRNd28w10CosineClassifierLocSelfsupervision.zip

$ cd ../../
```

**(2)** Download the model trained on the CIFAR-FS dataset.

```bash
# Run from the BF3S directory
$ mkdir ./experiments/cifar
$ cd ./experiments/cifar

# WRN-28-10-based CC with rotation prediction self-supervision model.
$ wget https://github.com/valeoai/BF3S/releases/download/0.2.0/WRNd28w10CosineClassifierRotAugRotSelfsupervision.zip
$ unzip WRNd28w10CosineClassifierRotAugRotSelfsupervision.zip

$ cd ../../
```

**(3)** Download the model trained on the ImageNet-FS dataset.

```bash
# Run from the BF3S directory
$ mkdir ./experiments/ImageNet
$ cd ./experiments/ImageNet

# ResNet10-based CC with rotation prediction self-supervision model.
$ wget https://github.com/valeoai/BF3S/releases/download/0.3.0/ResNet10CosineClassifierRotSelfsupervision.zip
$ unzip ResNet10CosineClassifierRotSelfsupervision.zip

$ cd ../../
```


## Train and test Cosine Classifier (CC) based few-shot models [3] with self-supervision.

### MiniImageNet: CC model with rotation prediction self-supervision.

To train and test the WRN-28-10 based Cosine Classifier model with
rotation prediction self-supervision run (you can skip the training step if you
have downloaded the pre-trained model):   
```bash
# Run from the BF3S directory
# Train the model.
$ python scripts/train_fewshot.py --config=miniImageNet/WRNd28w10CosineClassifierRotAugRotSelfsupervision

# Test the model on the 1-shot setting.
$ python scripts/test_fewshot.py --config=miniImageNet/WRNd28w10CosineClassifierRotAugRotSelfsupervision --num_novel=5 --num_train=1 --num_episodes=2000
# Expected 5-way classification accuracy: 62.81% with confidence interval +/- 0.46%

# Test the model on the 5-shot setting.
$ python scripts/test_fewshot.py --config=miniImageNet/WRNd28w10CosineClassifierRotAugRotSelfsupervision --num_novel=5 --num_train=5 --num_episodes=2000
# Expected 5-way classification accuracy: 80.00% with confidence interval +/- 0.34%
```

Note that the configuration file (of the above experiment) specified by the
`config` variable is located here:
`./config/miniImageNet/WRNd28w10CosineClassifierRotAugRotSelfsupervision.py`.
All the experiment configuration files are placed in the `./config/` directory.


### MiniImageNet: CC model with rotation prediction self-supervision and exploiting unlabeled images from tiered-MiniImageNet.

To train and test the WRN-28-10 based CC few-shot model with rotation
prediction self-supervision which also exploits (with the *semi-supervised 
learning* setting) unlabeled images from tiered-MiniImageNet, run (you can skip
the training step if you have downloaded the pre-trained model):
```bash
# Run from the BF3S directory
# Train the model.
$ python scripts/train_fewshot.py --config=miniImageNet/WRNd28w10CosineClassifierRotAugRotSelfsupervision_SemisupervisedTieredUnlabeled

# Test the model on the 1-shot setting.
$ python scripts/test_fewshot.py --config=miniImageNet/WRNd28w10CosineClassifierRotAugRotSelfsupervision_SemisupervisedTieredUnlabeled --num_novel=5 --num_train=1 --num_episodes=2000
# Expected 5-way classification accuracy: 64.03% with confidence interval +/- 0.46%

# Test the model on the 5-shot setting.
$ python scripts/test_fewshot.py --config=miniImageNet/WRNd28w10CosineClassifierRotAugRotSelfsupervision_SemisupervisedTieredUnlabeled --num_novel=5 --num_train=5 --num_episodes=2000
# Expected 5-way classification accuracy: 80.68% with confidence interval +/- 0.33%
```

### MiniImageNet: CC model with relative patch location prediction self-supervision.

To train and test the WRN-28-10 based Cosine Classifier few-shot model with
relative patch location prediction self-supervision run (you can skip 
the training step if you have downloaded the pre-trained model):
```bash
# Run from the BF3S directory
# Train the model.
$ python scripts/train_fewshot.py --config=miniImageNet/WRNd28w10CosineClassifierLocSelfsupervision

# Test the model on the 1-shot setting.
$ python scripts/test_fewshot.py --config=miniImageNet/WRNd28w10CosineClassifierLocSelfsupervision --num_novel=5 --num_train=1 --num_episodes=2000
# Expected 5-way classification accuracy: 60.70% with confidence interval +/- 0.47%

# Test the model on the 5-shot setting.
$ python scripts/test_fewshot.py --config=miniImageNet/WRNd28w10CosineClassifierLocSelfsupervision --num_novel=5 --num_train=5 --num_episodes=2000
# Expected 5-way classification accuracy: 77.61% with confidence interval +/- 0.33%
```

### CIFAR-FS: CC model with rotation prediction self-supervision.


To train and test the WRN-28-10 based Cosine Classifier few-shot model with
rotation prediction self-supervision and rotation augmentations run (you can
skip the training step if you have downloaded the pre-trained model):   
```bash
# Run from the BF3S directory
# Train the model.
$ python scripts/train_fewshot.py --config=cifar/WRNd28w10CosineClassifierRotAugRotSelfsupervision

# Test the model on the 1-shot setting.
$ python scripts/test_fewshot.py --config=cifar/WRNd28w10CosineClassifierRotAugRotSelfsupervision --num_novel=5 --num_train=1 --num_episodes=5000
# Expected 5-way classification accuracy: 75.38% with confidence interval +/- 0.31%

# Test the model on the 5-shot setting.
$ python scripts/test_fewshot.py --config=cifar/WRNd28w10CosineClassifierRotAugRotSelfsupervision --num_novel=5 --num_train=5 --num_episodes=5000
# Expected 5-way classification accuracy: 87.25% with confidence interval +/- 0.21%
```

### ImageNet-FS: CC model with rotation prediction self-supervision.

Instructions for training and testing the CC few-shot model with rotation
prediction self-supervision on the ImageNet based few-shot benchmark [1, 2].

**(1)** To train the ResNet10 based Cosine Classifier few-shot model
with rotation prediction self-supervision and rotation augmentations run:  
```bash
# Run from the BF3S directory
$ python scripts/train_fewshot.py --config=ImageNet/ResNet10CosineClassifierRotSelfsupervision --num_workers=8
```   
You can skip the above training step if you have downloaded the pre-trained
model.


**(2)** Extract and save the ResNet10 features (with the above model)
from images of the ImageNet dataset:    
```bash
# Run from the BF3S directory
# Extract features from the validation image split of the Imagenet.
$ python scripts/save_features_imagenet.py --config=ImageNet/ResNet10CosineClassifierRotSelfsupervision --split='val'
# Extract features from the training image split of the Imagenet.
$ python scripts/save_features_imagenet.py with config=ImageNet/ResNet10CosineClassifierRotSelfsupervision --split='train'
```   
The features will be saved on `./datasets/feature_datasets/ImageNet/ResNet10CosineClassifierRotSelfsupervision`.


**(4)** Test the model:   
```bash
# Run from the BF3S directory
# Test the CC+Rot model on the 1-shot setting.
$ python scripts/test_fewshot_imagenet.py --config=ImageNet/ResNet10CosineClassifierRotSelfsupervision_eval --testset --num_train=1 --bias_novel=0.8
# ==> Top 5 Accuracies:      [Novel: 46.43 | Base: 93.52 | All 57.88]

# Test the CC+Rot model on the 2-shot setting.
$ python scripts/test_fewshot_imagenet.py --config=ImageNet/ResNet10CosineClassifierRotSelfsupervision_eval --testset --num_train=2 --bias_novel=0.75
# ==> Top 5 Accuracies:      [Novel: 57.80 | Base: 93.52 | All 64.76]

# Test the CC+Rot model on the 5-shot setting.
$ python scripts/test_fewshot_imagenet.py --config=ImageNet/ResNet10CosineClassifierRotSelfsupervision_eval --testset --num_train=5 --bias_novel=0.7
# ==> Top 5 Accuracies:      [Novel: 69.67 | Base: 93.52 | All 72.29]

# Test the CC+Rot model on the 10-shot setting.
$ python scripts/test_fewshot_imagenet.py --config=ImageNet/ResNet10CosineClassifierRotSelfsupervision_eval --testset --num_train=10 --bias_novel=0.65
# ==> Top 5 Accuracies:      [Novel: 74.64 | Base: 93.52 | All 75.63]

# Test the CC+Rot model on the 20-shot setting.
$ python scripts/test_fewshot_imagenet.py --config=ImageNet/ResNet10CosineClassifierRotSelfsupervision_eval --testset --num_train=20 --bias_novel=0.6
# ==> Top 5 Accuracies:      [Novel: 77.31 | Base: 93.52 | All 77.40]
```   
Note that here, to evaluate the model trained with the
`./config/ImageNet/ResNet10CosineClassifierRotSelfsupervision.py` config file,
we used a different config file named
`./config/ImageNet/ResNet10CosineClassifierRotSelfsupervision_eval.py`.
Also, the `--bias_novel` term specifies a multiplicative bias for the
classification scores of the novel classes. Its purpose is to balance the
classification scores of the base and novel classes (necessary since
the classifiers for those two different classes are trained in different ways
and different stages). It only affects the All classification accuracy metrics.
The used bias values were tuned on the validation split.


### References
```
[1] B. Hariharan and R. Girshick. Low-shot visual recognition by shrinking and hallucinating features.
[2] Y.-X. Wang and R. Girshick, M. Hebert, B. Hariharan. Low-shot learning from imaginary data.
[3] S. Gidaris and N. Komodakis. Dynamic few-shot visual learning without forgetting.
```
