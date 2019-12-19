import torch
import torch.nn.functional as F

import bf3s.algorithms.classification.utils as cls_utils
import bf3s.algorithms.fewshot.utils as fewshot_utils
import bf3s.utils as utils

_CENTRAL_PATCH_INDEX = 4
_NUM_OF_PATCHES = 9
_NUM_LOCATION_CLASSES = _NUM_OF_PATCHES - 1


def generate_patch_locations():
    """Generates patch locations."""
    locations = [i for i in range(_NUM_OF_PATCHES) if i != _CENTRAL_PATCH_INDEX]
    assert len(locations) == _NUM_LOCATION_CLASSES
    return _CENTRAL_PATCH_INDEX, locations


def generate_location_labels(batch_size, is_cuda):
    """Generates location prediction labels."""
    location_labels = torch.arange(_NUM_LOCATION_CLASSES).view(1, _NUM_LOCATION_CLASSES)
    if is_cuda:
        location_labels = location_labels.to("cuda")
    location_labels = location_labels.repeat(batch_size, 1).view(-1)
    return location_labels


def add_patch_dimension(patches):
    """Add the patch dimension to a mini-batch of patches."""
    assert (patches.size(0) % _NUM_OF_PATCHES) == 0
    return utils.add_dimension(patches, dim_size=(patches.size(0) // _NUM_OF_PATCHES))


def concatenate_accross_channels_patches(patches):
    assert patches.dim() == 3 or patches.dim() == 5
    if patches.dim() == 3:
        batch_size, _, channels = patches.size()
        return patches.view(batch_size, _NUM_OF_PATCHES * channels)
    elif patches.dim() == 5:
        batch_size, _, channels, height, width = patches.size()
        return patches.view(batch_size, _NUM_OF_PATCHES * channels, height, width)


def create_patch_pairs(patches, central, locations):
    """Creates patch pairs."""
    assert patches.size(1) == 9
    num_dims = patches.dim()
    if num_dims == 3:
        patches = patches.view(patches.size(0), 9, patches.size(2), 1, 1)

    assert patches.dim() == 5
    batch_size, _, channels, height, width = patches.size()

    patches_central = patches[:, central, :, :, :]
    patch_pairs = []
    for loc in locations:
        patches_loc = patches[:, loc, :, :, :]
        patch_pairs.append(torch.cat([patches_loc, patches_central], dim=1))

    patch_pairs = torch.stack(patch_pairs, dim=1)
    if num_dims == 3:
        patch_pairs = patch_pairs.view(batch_size * len(locations), 2 * channels)
    else:
        patch_pairs = patch_pairs.view(
            batch_size * len(locations), 2 * channels, height, width
        )

    return patch_pairs


def patch_location_task(location_classifier, features):
    """Applies the patch location prediction head to the given features."""
    features = add_patch_dimension(features)
    batch_size = features.size(0)

    central, locations = generate_patch_locations()
    location_labels = generate_location_labels(batch_size, features.is_cuda)
    features_pairs = create_patch_pairs(features, central, locations)
    scores = location_classifier(features_pairs)
    assert scores.size(1) == _NUM_LOCATION_CLASSES
    loss = F.cross_entropy(scores, location_labels)

    return scores, loss, location_labels


def combine_multiple_patch_features(features, combine):
    """Combines the multiple patches of an image."""
    if combine == "average":
        return features.mean(dim=1)
    elif combine == "concatenate":
        return concatenate_accross_channels_patches(features)
    else:
        raise ValueError(f"Not supported combine option {combine}")


def patch_classification_task(patch_classifier, features, labels, combine):
    """Applies the auxiliary task of classifying individual patches."""
    features = add_patch_dimension(features)
    features = combine_multiple_patch_features(features, combine)

    scores = patch_classifier(features)
    loss = F.cross_entropy(scores, labels)
    return scores, loss


def object_classification_with_patch_location_selfsupervision(
    feature_extractor,
    feature_extractor_optimizer,
    classifier,
    classifier_optimizer,
    location_classifier,
    location_classifier_optimizer,
    patch_classifier,
    patch_classifier_optimizer,
    images,
    labels,
    patches,
    labels_patches,
    is_train,
    patch_location_loss_coef=1.0,
    patch_classification_loss_coef=1.0,
    combine="average",
    base_ids=None,
    standardize_patches=True,
):
    """Forward-backward propagation routine for classification model extended
    with the auxiliary self-supervised task of predicting the relative location
    of patches."""

    if base_ids is not None:
        assert base_ids.size(0) == 1

    assert images.dim() == 4
    assert labels.dim() == 1
    assert images.size(0) == labels.size(0)

    assert patches.dim() == 5 and patches.size(1) == 9
    assert patches.size(0) == labels_patches.size(0)
    patches = utils.convert_from_5d_to_4d(patches)
    if standardize_patches:
        patches = utils.standardize_image(patches)

    if is_train:
        # Zero gradients.
        feature_extractor_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        if patch_location_loss_coef > 0.0:
            location_classifier_optimizer.zero_grad()
        if patch_classification_loss_coef > 0.0:
            patch_classifier_optimizer.zero_grad()

    record = {}
    with torch.set_grad_enabled(is_train):
        # Extract features from the images.
        features_images = feature_extractor(images)
        # Extract features from the image patches.
        features_patches = feature_extractor(patches)

        # Perform object classification task.
        scores_classification, loss_classsification = cls_utils.classification_task(
            classifier, features_images, labels, base_ids
        )
        record["loss_cls"] = loss_classsification.item()
        loss_total = loss_classsification

        # Perform the self-supervised task of relative patch locatioon
        # prediction.
        if patch_location_loss_coef > 0.0:
            scores_location, loss_location, labels_loc = patch_location_task(
                location_classifier, features_patches
            )
            record["loss_loc"] = loss_location.item()
            loss_total = loss_total + loss_location * patch_location_loss_coef

        # Perform the auxiliary task of classifying individual patches.
        if patch_classification_loss_coef > 0.0:
            scores_patch, loss_patch = patch_classification_task(
                patch_classifier, features_patches, labels_patches, combine
            )
            record["loss_patch_cls"] = loss_patch.item()
            loss_total = loss_total + loss_patch * patch_classification_loss_coef

        # Because the total loss consists of multiple individual losses
        # (i.e., 3) scale it down by a factor of 0.5.
        loss_total = loss_total * 0.5

    with torch.no_grad():
        # Compute accuracies.
        record["Accuracy"] = utils.top1accuracy(scores_classification, labels)
        if patch_location_loss_coef > 0.0:
            record["AccuracyLoc"] = utils.top1accuracy(scores_location, labels_loc)
        if patch_classification_loss_coef > 0.0:
            record["AccuracyPatch"] = utils.top1accuracy(scores_patch, labels_patches)

    if is_train:
        # Backward loss and apply gradient steps.
        loss_total.backward()
        feature_extractor_optimizer.step()
        classifier_optimizer.step()
        if patch_location_loss_coef > 0.0:
            location_classifier_optimizer.step()
        if patch_classification_loss_coef > 0.0:
            patch_classifier_optimizer.step()

    return record


def fewshot_classification_with_patch_location_selfsupervision(
    feature_extractor,
    feature_extractor_optimizer,
    classifier,
    classifier_optimizer,
    location_classifier,
    location_classifier_optimizer,
    patch_classifier,
    patch_classifier_optimizer,
    images_train,
    patches_train,
    labels_train,
    labels_train_1hot,
    images_test,
    patches_test,
    labels_test,
    is_train,
    base_ids=None,
    patch_location_loss_coef=1.0,
    patch_classification_loss_coef=1.0,
    combine="average",
    standardize_patches=True,
):
    """Forward-backward propagation routine for few-shot model extended
    with the auxiliary self-supervised task of predicting the relative location
    of patches."""

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

    assert patches_train.dim() == 6
    assert patches_train.size(0) == images_train.size(0)
    assert patches_train.size(1) == images_train.size(1)
    assert patches_train.size(2) == 9

    assert patches_test.dim() == 6
    assert patches_test.size(0) == images_test.size(0)
    assert patches_test.size(1) == images_test.size(1)
    assert patches_test.size(2) == 9

    meta_batch_size = images_train.size(0)
    num_train = images_train.size(1)
    num_test = images_test.size(1)

    if is_train:
        # Zero gradients.
        feature_extractor_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        if patch_location_loss_coef > 0.0:
            location_classifier_optimizer.zero_grad()
        if patch_classification_loss_coef > 0.0:
            patch_classifier_optimizer.zero_grad()

    record = {}
    with torch.no_grad():
        images_train = utils.convert_from_5d_to_4d(images_train)
        images_test = utils.convert_from_5d_to_4d(images_test)
        labels_test = labels_test.view(-1)
        images = torch.cat([images_train, images_test], dim=0)

        batch_size_train = images_train.size(0)
        batch_size_train_test = images.size(0)
        assert batch_size_train == meta_batch_size * num_train
        assert batch_size_train_test == meta_batch_size * (num_train + num_test)

        patches_train = utils.convert_from_6d_to_4d(patches_train)
        patches_test = utils.convert_from_6d_to_4d(patches_test)
        if standardize_patches:
            patches_train = utils.standardize_image(patches_train)
            patches_test = utils.standardize_image(patches_test)
        patches = torch.cat([patches_train, patches_test], dim=0)

        assert patches_train.size(0) == batch_size_train * 9
        assert patches.size(0) == batch_size_train_test * 9

    with torch.set_grad_enabled(is_train):
        # Extract features from the images.
        features = feature_extractor(images)
        # Extract features from the image patches.
        features_patches = feature_extractor(patches)

        # Perform object classification task.
        features_train = features[:batch_size_train]
        features_test = features[batch_size_train:batch_size_train_test]
        features_train = utils.add_dimension(features_train, meta_batch_size)
        features_test = utils.add_dimension(features_test, meta_batch_size)
        (
            classification_scores,
            loss_classsification,
        ) = fewshot_utils.few_shot_feature_classification(
            classifier,
            features_test,
            features_train,
            labels_train_1hot,
            labels_test,
            base_ids,
        )
        record["loss_cls"] = loss_classsification.item()
        loss_total = loss_classsification

        # Perform the self-supervised task of relative patch locatioon
        # prediction.
        if patch_location_loss_coef > 0.0:
            scores_location, loss_location, labels_loc = patch_location_task(
                location_classifier, features_patches
            )
            record["loss_loc"] = loss_location.item()
            loss_total = loss_total + loss_location * patch_location_loss_coef

        # Perform the auxiliary task of classifying patches.
        if patch_classification_loss_coef > 0.0:
            features_patches = add_patch_dimension(features_patches)
            assert features_patches.size(0) == batch_size_train_test
            assert features_patches.size(1) == 9
            features_patches = combine_multiple_patch_features(features_patches, combine)

            features_patches_train = utils.add_dimension(
                features_patches[:batch_size_train], meta_batch_size
            )
            features_patches_test = utils.add_dimension(
                features_patches[batch_size_train:batch_size_train_test], meta_batch_size
            )

            scores_patch, loss_patch = fewshot_utils.few_shot_feature_classification(
                patch_classifier,
                features_patches_test,
                features_patches_train,
                labels_train_1hot,
                labels_test,
                base_ids,
            )
            record["loss_patch_cls"] = loss_patch.item()
            loss_total = loss_total + loss_patch * patch_classification_loss_coef

        # Because the total loss consists of multiple individual losses
        # (i.e., 3) scale it down by a factor of 0.5.
        loss_total = loss_total * 0.5

    with torch.no_grad():
        num_base = base_ids.size(1) if (base_ids is not None) else 0
        record = fewshot_utils.compute_accuracy_metrics(
            classification_scores, labels_test, num_base, record
        )
        if patch_location_loss_coef > 0.0:
            record["AccuracyLoc"] = utils.top1accuracy(scores_location, labels_loc)
        if patch_classification_loss_coef > 0.0:
            record = fewshot_utils.compute_accuracy_metrics(
                scores_patch, labels_test, num_base, record, string_id="Patch"
            )

    if is_train:
        # Backward loss and apply gradient steps.
        loss_total.backward()
        feature_extractor_optimizer.step()
        classifier_optimizer.step()
        if patch_location_loss_coef > 0.0:
            location_classifier_optimizer.step()
        if patch_classification_loss_coef > 0.0:
            patch_classifier_optimizer.step()

    return record
