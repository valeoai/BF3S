import torch
import torch.nn.functional as F

import bf3s.utils as utils


def extract_features(feature_extractor, images, feature_name=None):
    """Extracts features from images using the provided feature extractor."""
    if feature_name:
        # Extract the features of the feature levels specified by feature_name.
        if isinstance(feature_name, str):
            feature_name = [
                feature_name,
            ]
        assert isinstance(feature_name, (list, tuple))

        features = feature_extractor(images, out_feat_keys=feature_name)
    else:
        # Extract the features from the last feature level.
        features = feature_extractor(images)

    return features


def classification_task(classifier, features, labels, base_ids=None):
    """Applies the classifier to features and computes the classifcation loss"""

    if base_ids is not None:
        assert base_ids.dim() == 2
        batch_size = features.size(0)
        meta_batch_size = base_ids.size(0)
        features = utils.add_dimension(features, dim_size=meta_batch_size)
        scores = classifier(features_test=features, base_ids=base_ids)
        scores = scores.view(batch_size, -1)
    else:
        scores = classifier(features)

    loss = F.cross_entropy(scores, labels)

    return scores, loss


def object_classification(
    feature_extractor,
    feature_extractor_optimizer,
    classifier,
    classifier_optimizer,
    images,
    labels,
    is_train,
    base_ids=None,
    feature_name=None,
):
    """Forward-backward propagation routine for the classification task.

    Given as input a mini-batch of images with their labels, it applies the
    forward and (optionaly) backward propagation routines of the classification
    task. The code assumes that the classification model is divided into a
    feature extractor network and a classification head network.

    Args:
    feature_extractor: The feature extractor neural network.
    feature_extractor_optimizer: The parameter optimizer of the feature
        extractor. If None, then the feature extractor remains frozen during
        training.
    classifier: The classification head applied on the output of the feature
        extractor.
    classifier_optimizer: The parameter optimizer of the classification head.
    images: A 4D tensor of shape [batch_size x channels x height x width] with
        the mini-batch images. It is assumed that this tensor is already on the
        same device as the feature extractor and classification head networks.
    labels: A 1D tensor with shape [batch_size] with the image labels. It is
        assumed that this tensor is already on the same device as the feature
        extractor and classification head networks.
    is_train: Boolean value that indicates if this mini-batch of images will be
        used for training or testing. If is_train is False, then the code does
        not apply the backward propagation step and does not update the
        parameter optimizers.
    base_ids: Optional argument used in case of episodic training of few-shot
        classification models. In this case, it is assumed that the total input
        batch_size consists of meta_batch_size training episodes, each with
        (batch_size // meta_batch_size) inner batch size (i.e., it must hold
        that batch_size % meta_batch_size == 0). In this context, base_ids is a
        2D tensor with shape [meta_batch_size x num_base], where base_ids[m] are
        the indices of the base categories that are being used in the m-th
        training episode.
    feature_name: (optional) A string or list of strings with the name of
        feature level(s) from which the feature extractor will extract features
        for the classification task.

    Returns:
    record: A dictionary of scalar values with the following items:
        'loss': The cross entropy loss of this mini-batch of images.
        'AccuracyTop1': The top-1 classification accuracy.
    """

    assert not (isinstance(feature_name, (list, tuple)) and len(feature_name) > 1)

    assert labels.dim() == 1
    assert images.size(0) == labels.size(0)

    if is_train:  # Zero gradients.
        if feature_extractor_optimizer:
            feature_extractor_optimizer.zero_grad()
        classifier_optimizer.zero_grad()

    record = {}
    train_feature_extractor = is_train and (feature_extractor_optimizer is not None)
    with torch.set_grad_enabled(train_feature_extractor):
        # Extract features from the images.
        features = extract_features(feature_extractor, images, feature_name=feature_name)

    if not train_feature_extractor:
        # Make sure that no gradients are backproagated to the feature
        # extractor when the feature extraction model is freezed.
        features = features.detach()

    with torch.set_grad_enabled(is_train):
        # Perform the object classification task.
        scores_classification, loss_classsification = classification_task(
            classifier, features, labels, base_ids
        )
        loss_total = loss_classsification
        record["loss"] = loss_total.item()

    with torch.no_grad():  # Compute accuracies.
        record["AccuracyTop1"] = utils.top1accuracy(scores_classification, labels)

    if is_train:  # Backward loss and apply gradient steps.
        loss_total.backward()
        if feature_extractor_optimizer:
            feature_extractor_optimizer.step()
        classifier_optimizer.step()

    return record
