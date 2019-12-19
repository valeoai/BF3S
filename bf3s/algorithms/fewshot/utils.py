import numpy as np
import torch
import torch.nn.functional as F

import bf3s.algorithms.classification.utils as cls_utils
import bf3s.utils as utils


def few_shot_feature_classification(
    classifier, features_test, features_train, labels_train_1hot, labels_test, base_ids=None
):
    """Applies the classification head of few-shot classification model."""

    if base_ids is not None:
        classification_scores = classifier(
            features_test=features_test,
            features_train=features_train,
            labels_train=labels_train_1hot,
            base_ids=base_ids,
        )
    else:
        classification_scores = classifier(
            features_test=features_test,
            features_train=features_train,
            labels_train=labels_train_1hot,
        )

    assert classification_scores.dim() == 3

    classification_scores = classification_scores.view(
        classification_scores.size(0) * classification_scores.size(1), -1
    )
    labels_test = labels_test.view(-1)
    assert classification_scores.size(0) == labels_test.size(0)

    loss = F.cross_entropy(classification_scores, labels_test)

    return classification_scores, loss


def compute_accuracy_metrics(scores, labels, num_base, record={}, string_id=""):
    """Computes the classification accuracies of a mini-batch of episodes."""
    assert isinstance(record, dict)

    if string_id != "":
        string_id = "_" + string_id

    if labels.dim() > 1:
        labels = labels.view(scores.size(0))

    if num_base > 0:
        record["AccuracyBoth" + string_id] = utils.top1accuracy(scores, labels)

        base_indices = torch.nonzero(labels < num_base).view(-1)
        novel_indices = torch.nonzero(labels >= num_base).view(-1)
        if base_indices.dim() != 0 and base_indices.size(0) > 0:
            scores_base = scores[base_indices][:, :num_base]
            labels_base = labels[base_indices]
            record["AccuracyBase" + string_id] = utils.top1accuracy(scores_base, labels_base)

        scores_novel = scores[novel_indices, :][:, num_base:]
        labels_novel = labels[novel_indices] - num_base
        record["AccuracyNovel" + string_id] = utils.top1accuracy(scores_novel, labels_novel)
    else:
        record["AccuracyNovel" + string_id] = utils.top1accuracy(scores, labels)

    return record


def fewshot_classification(
    feature_extractor,
    feature_extractor_optimizer,
    classifier,
    classifier_optimizer,
    images_train,
    labels_train,
    labels_train_1hot,
    images_test,
    labels_test,
    is_train,
    base_ids=None,
    feature_name=None,
    classification_coef=1.0,
):
    """Forward-backward propagation routine of the few-shot classification task.

    Given as input a mini-batch of few-shot episodes, it applies the
    forward and (optionally) backward propagation routines of the few-shot
    classification task. Each episode consists of (1) num_train_examples number
    of training examples for the novel classes of the few-shot episode, (2) the
    labels of training examples of the novel classes, (3) num_test_examples
    number of test examples of the few-shot episode (note that the test
    examples can be from both base and novel classes), and (4) the labels of the
    test examples. Each mini-batch consists of meta_batch_size number of
    few-shot episodes. The code assumes that the few-shot classification model
    is divided into a feature extractor network and a classification head
    network.

    Args:
    feature_extractor: The feature extractor neural network.
    feature_extractor_optimizer: The parameter optimizer of the feature
        extractor. If None, then the feature extractor remains frozen during
        training.
    classifier: The classification head applied on the output of the feature
        extractor.
    classifier_optimizer: The parameter optimizer of the classification head.
    images_train: A 5D tensor with shape
        [meta_batch_size x num_train_examples x channels x height x width] that
        represents a mini-batch of meta_batch_size number of few-shot episodes,
        each with num_train_examples number of training examples.
    labels_train: A 2D tensor with shape
        [meta_batch_size x num_train_examples] that represents the discrete
        labels of the training examples of each few-shot episode in the batch.
    labels_train_1hot: A 3D tensor with shape
        [meta_batch_size x num_train_examples x num_novel] that represents
        the 1hot labels of the training examples of the novel classes of each
        few-shot episode in the batch. num_novel is the number of novel classes
        per few-shot episode.
    images_test: A 5D tensor with shape
        [meta_batch_size x num_test_examples x channels x height x width] that
        represents a mini-batch of meta_batch_size number of few-shot episodes,
        each with num_test_examples number of test examples.
    labels_test: A 2D tensor with shape
        [meta_batch_size x num_test_examples] that represents the discrete
        labels of the test examples of each few-shot episode in the mini-batch.
    is_train: Boolean value that indicates if this mini-batch will be
        used for training or testing. If is_train is False, then the code does
        not apply the backward propagation step and does not update the
        parameter optimizers.
    base_ids: A 2D tensor with shape [meta_batch_size x num_base], where
        base_ids[m] are the indices of the base categories that are being used
        in the m-th few-shot episode. num_base is the number of base classes per
        few-shot episode.
    feature_name: (optional) A string or list of strings with the name of
        feature level(s) from which the feature extractor will extract features
        for the classification task.
    classification_coef: (optional) the loss weight of the few-shot
        classification task.

    Returns:
    record: A dictionary of scalar values with the following items:
        'loss': The cross entropy loss of this mini-batch.
        'AccuracyNovel': The classification accuracy of the test examples among
            only the novel classes.
        'AccuracyBase': (optinional) The classification accuracy of the test
            examples among only the base classes. Applicable, only if there are
            test examples from base classes in the mini-batch.
        'AccuracyBase': (optinional) The classification accuracy of the test
            examples among both the base and novel classes. Applicable, only if
            there are test examples from base classes in the mini-batch.
    """

    assert images_train.dim() == 5
    assert images_test.dim() == 5
    assert images_train.size(0) == images_test.size(0)
    assert images_train.size(2) == images_test.size(2)
    assert images_train.size(3) == images_test.size(3)
    assert images_train.size(4) == images_test.size(4)
    assert labels_train.dim() == 2
    assert labels_test.dim() == 2
    assert labels_train.size(0) == labels_test.size(0)
    assert labels_train.size(0) == images_train.size(0)

    assert not (isinstance(feature_name, (list, tuple)) and len(feature_name) > 1)

    meta_batch_size = images_train.size(0)

    if is_train:  # zero the gradients
        if feature_extractor_optimizer:
            feature_extractor_optimizer.zero_grad()
        classifier_optimizer.zero_grad()

    record = {}
    with torch.no_grad():
        images_train = utils.convert_from_5d_to_4d(images_train)
        images_test = utils.convert_from_5d_to_4d(images_test)
        labels_test = labels_test.view(-1)
        batch_size_train = images_train.size(0)
        # batch_size_test = images_test.size(0)
        images = torch.cat([images_train, images_test], dim=0)

    train_feature_extractor = is_train and (feature_extractor_optimizer is not None)
    with torch.set_grad_enabled(train_feature_extractor):
        # Extract features from the train and test images.
        features = cls_utils.extract_features(
            feature_extractor, images, feature_name=feature_name
        )

    if not train_feature_extractor:
        # Make sure that no gradients are backproagated to the feature
        # extractor when the feature extraction model is freezed.
        features = features.detach()

    with torch.set_grad_enabled(is_train):
        features_train = features[:batch_size_train]
        features_test = features[batch_size_train:]
        features_train = utils.add_dimension(features_train, meta_batch_size)
        features_test = utils.add_dimension(features_test, meta_batch_size)

        # Apply the classification head of the few-shot classification model.
        classification_scores, loss = few_shot_feature_classification(
            classifier,
            features_test,
            features_train,
            labels_train_1hot,
            labels_test,
            base_ids,
        )
        record["loss"] = loss.item()
        loss_total = loss * classification_coef
        # *******************************************************************

    with torch.no_grad():
        num_base = base_ids.size(1) if (base_ids is not None) else 0
        record = compute_accuracy_metrics(classification_scores, labels_test, num_base, record)

    if is_train:
        loss_total.backward()
        if feature_extractor_optimizer:
            feature_extractor_optimizer.step()
        classifier_optimizer.step()

    return record


def compute_95confidence_intervals(
    record, episode, num_episodes, store_accuracies, metrics=["AccuracyNovel",]
):
    """Computes the 95% confidence interval for the novel class accuracy."""

    if episode == 0:
        store_accuracies = {metric: [] for metric in metrics}

    for metric in metrics:
        store_accuracies[metric].append(record[metric])
        if episode == (num_episodes - 1):
            # Compute std and confidence interval of the 'metric' accuracies.
            accuracies = np.array(store_accuracies[metric])
            stds = np.std(accuracies, 0)
            record[metric + "_std"] = stds
            record[metric + "_cnf"] = 1.96 * stds / np.sqrt(num_episodes)

    return record, store_accuracies
