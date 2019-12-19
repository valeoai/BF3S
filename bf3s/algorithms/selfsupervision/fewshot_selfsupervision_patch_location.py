import torch

import bf3s.algorithms.algorithm as algorithm
import bf3s.algorithms.classification.utils as cls_utils
import bf3s.algorithms.fewshot.utils as fs_utils
import bf3s.algorithms.selfsupervision.patch_location_utils as loc_utils
import bf3s.utils as utils


class FewShotPatchLocationSelfSupervision(algorithm.Algorithm):
    """Trains a few-shot model with the auxiliary location prediction task."""

    def __init__(self, opt, _run=None, _log=None):
        super().__init__(opt, _run, _log)
        self.keep_best_model_metric_name = "AccuracyNovel"
        self.patch_location_loss_coef = opt["patch_location_loss_coef"]
        self.patch_classification_loss_coef = opt["patch_classification_loss_coef"]
        self.standardize_image = opt["standardize_image"]
        self.combine_patches = opt["combine_patches"]
        self.standardize_patches = opt["standardize_patches"]

        self.accuracies = {}

    def allocate_tensors(self):
        self.tensors = {
            "images_train": torch.FloatTensor(),
            "labels_train": torch.LongTensor(),
            "labels_train_1hot": torch.FloatTensor(),
            "images_test": torch.FloatTensor(),
            "labels_test": torch.LongTensor(),
            "Kids": torch.LongTensor(),
            "patches_train": torch.FloatTensor(),
            "patches_test": torch.FloatTensor(),
            "patches": torch.FloatTensor(),
            "labels_patches": torch.LongTensor(),
        }

    def set_tensors(self, batch):
        two_datasets = (
            isinstance(batch, (list, tuple))
            and len(batch) == 2
            and isinstance(batch[0], (list, tuple))
            and isinstance(batch[1], (list, tuple))
        )

        if two_datasets:
            train_test_stage = "classification"
            assert len(batch[0]) == 4
            assert len(batch[1]) == 2

            images_test, labels_test, K, num_base_per_episode = batch[0]
            patches, labels_patches = batch[1]

            self.num_base = num_base_per_episode[0].item()
            self.tensors["images_test"].resize_(images_test.size()).copy_(images_test)
            self.tensors["labels_test"].resize_(labels_test.size()).copy_(labels_test)
            self.tensors["Kids"].resize_(K.size()).copy_(K)

            self.tensors["patches"].resize_(patches.size()).copy_(patches)
            self.tensors["labels_patches"].resize_(labels_patches.size()).copy_(labels_patches)

        elif len(batch) == 6:
            train_test_stage = "fewshot"
            (
                images_train,
                labels_train,
                images_test,
                labels_test,
                K,
                num_base_per_episode,
            ) = batch
            self.num_base = num_base_per_episode[0].item()
            self.tensors["images_train"].resize_(images_train.size()).copy_(images_train)
            self.tensors["labels_train"].resize_(labels_train.size()).copy_(labels_train)
            labels_train = self.tensors["labels_train"]

            nKnovel = 1 + labels_train.max().item() - self.num_base

            labels_train_1hot_size = list(labels_train.size()) + [
                nKnovel,
            ]
            labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
            self.tensors["labels_train_1hot"].resize_(labels_train_1hot_size).fill_(
                0
            ).scatter_(
                len(labels_train_1hot_size) - 1, labels_train_unsqueeze - self.num_base, 1
            )
            self.tensors["images_test"].resize_(images_test.size()).copy_(images_test)
            self.tensors["labels_test"].resize_(labels_test.size()).copy_(labels_test)
            self.tensors["Kids"].resize_(K.size()).copy_(K)

        elif len(batch) == 8:
            train_test_stage = "fewshot"
            (
                images_train,
                patches_train,
                labels_train,
                images_test,
                patches_test,
                labels_test,
                K,
                num_base_per_episode,
            ) = batch

            self.num_base = num_base_per_episode[0].item()
            self.tensors["images_train"].resize_(images_train.size()).copy_(images_train)
            self.tensors["patches_train"].resize_(patches_train.size()).copy_(patches_train)
            self.tensors["labels_train"].resize_(labels_train.size()).copy_(labels_train)
            labels_train = self.tensors["labels_train"]

            nKnovel = 1 + labels_train.max().item() - self.num_base

            labels_train_1hot_size = list(labels_train.size()) + [
                nKnovel,
            ]
            labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
            self.tensors["labels_train_1hot"].resize_(labels_train_1hot_size).fill_(
                0
            ).scatter_(
                len(labels_train_1hot_size) - 1, labels_train_unsqueeze - self.num_base, 1
            )
            self.tensors["images_test"].resize_(images_test.size()).copy_(images_test)
            self.tensors["patches_test"].resize_(patches_test.size()).copy_(patches_test)
            self.tensors["labels_test"].resize_(labels_test.size()).copy_(labels_test)
            self.tensors["Kids"].resize_(K.size()).copy_(K)

        elif len(batch) == 4:
            train_test_stage = "classification"
            images_test, labels_test, K, num_base_per_episode = batch

            self.num_base = num_base_per_episode[0].item()
            self.tensors["images_test"].resize_(images_test.size()).copy_(images_test)
            self.tensors["labels_test"].resize_(labels_test.size()).copy_(labels_test)
            self.tensors["Kids"].resize_(K.size()).copy_(K)
        elif len(batch) == 5:
            train_test_stage = "classification"
            images_test, patches_test, labels_test, K, num_base_per_episode = batch

            self.num_base = num_base_per_episode[0].item()
            self.tensors["images_test"].resize_(images_test.size()).copy_(images_test)
            self.tensors["labels_test"].resize_(labels_test.size()).copy_(labels_test)
            self.tensors["Kids"].resize_(K.size()).copy_(K)

            patches_test = patches_test.view(
                patches_test.size(0) * patches_test.size(1),  # 1 * 64
                patches_test.size(2),  # 9
                patches_test.size(3),  # 3
                patches_test.size(4),  # 24
                patches_test.size(5),
            )  # 24
            labels_patches = labels_test.view(-1)

            self.tensors["patches"].resize_(patches_test.size()).copy_(patches_test)
            self.tensors["labels_patches"].resize_(labels_patches.size()).copy_(labels_patches)

        return train_test_stage

    def train_step(self, batch):
        return self.process_batch(batch, is_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, is_train=False)

    def process_batch(self, batch, is_train):
        process_type = self.set_tensors(batch)
        if process_type == "fewshot":
            record = self.process_batch_fewshot_classification_task(is_train)
        elif process_type == "classification":
            record = self.process_batch_base_class_classification_task(is_train)
        else:
            raise ValueError(f"Unexpected process type {process_type}")

        return record

    def process_batch_base_class_classification_task(self, is_train):

        images = self.tensors["images_test"]
        labels = self.tensors["labels_test"]
        Kids = self.tensors["Kids"]
        base_ids = Kids[:, : self.num_base].contiguous()
        assert images.dim() == 5 and labels.dim() == 2
        images = utils.convert_from_5d_to_4d(images)

        if self.standardize_image:
            images = utils.standardize_image(images)

        labels = labels.view(-1)

        patches = self.tensors["patches"]
        labels_patches = self.tensors["labels_patches"]

        auxiliary_tasks = is_train and (
            self.patch_location_loss_coef > 0.0 or self.patch_classification_loss_coef > 0.0
        )

        if auxiliary_tasks:
            record = loc_utils.object_classification_with_patch_location_selfsupervision(
                feature_extractor=self.networks["feature_extractor"],
                feature_extractor_optimizer=self.optimizers["feature_extractor"],
                classifier=self.networks["classifier"],
                classifier_optimizer=self.optimizers["classifier"],
                location_classifier=self.networks.get("classifier_loc"),
                location_classifier_optimizer=self.optimizers.get("classifier_loc"),
                patch_classifier=self.networks.get("patch_classifier"),
                patch_classifier_optimizer=self.optimizers.get("patch_classifier"),
                images=images,
                labels=labels,
                patches=patches,
                labels_patches=labels_patches,
                is_train=is_train,
                patch_location_loss_coef=self.patch_location_loss_coef,
                patch_classification_loss_coef=self.patch_classification_loss_coef,
                combine=self.combine_patches,
                base_ids=base_ids,
                standardize_patches=self.standardize_patches,
            )
        else:
            record = cls_utils.object_classification(
                feature_extractor=self.networks["feature_extractor"],
                feature_extractor_optimizer=self.optimizers["feature_extractor"],
                classifier=self.networks["classifier"],
                classifier_optimizer=self.optimizers["classifier"],
                images=images,
                labels=labels,
                is_train=is_train,
                base_ids=base_ids,
            )

        return record

    def process_batch_fewshot_classification_task(self, is_train):
        Kids = self.tensors["Kids"]
        base_ids = None if (self.num_base == 0) else Kids[:, : self.num_base].contiguous()

        images_train = self.tensors["images_train"]
        images_test = self.tensors["images_test"]

        if self.standardize_image:
            assert images_train.dim() == 5 and images_test.dim() == 5
            assert images_train.size(0) == images_test.size(0)
            meta_batch_size = images_train.size(0)
            images_train = utils.convert_from_5d_to_4d(images_train)
            images_test = utils.convert_from_5d_to_4d(images_test)

            images_train = utils.standardize_image(images_train)
            images_test = utils.standardize_image(images_test)

            images_train = utils.add_dimension(images_train, meta_batch_size)
            images_test = utils.add_dimension(images_test, meta_batch_size)

        auxiliary_tasks = is_train and (
            self.patch_location_loss_coef > 0.0 or self.patch_classification_loss_coef > 0.0
        )

        if auxiliary_tasks:
            record = loc_utils.fewshot_classification_with_patch_location_selfsupervision(
                feature_extractor=self.networks["feature_extractor"],
                feature_extractor_optimizer=self.optimizers["feature_extractor"],
                classifier=self.networks["classifier"],
                classifier_optimizer=self.optimizers["classifier"],
                location_classifier=self.networks.get("classifier_loc"),
                location_classifier_optimizer=self.optimizers.get("classifier_loc"),
                patch_classifier=self.networks.get("patch_classifier"),
                patch_classifier_optimizer=self.optimizers.get("patch_classifier"),
                images_train=images_train,
                patches_train=self.tensors["patches_train"],
                labels_train=self.tensors["labels_train"],
                labels_train_1hot=self.tensors["labels_train_1hot"],
                images_test=images_test,
                patches_test=self.tensors["patches_test"],
                labels_test=self.tensors["labels_test"],
                is_train=is_train,
                base_ids=base_ids,
                patch_location_loss_coef=self.patch_location_loss_coef,
                patch_classification_loss_coef=self.patch_classification_loss_coef,
                combine=self.combine_patches,
                standardize_patches=self.standardize_patches,
            )
        else:
            record = fs_utils.fewshot_classification(
                feature_extractor=self.networks["feature_extractor"],
                feature_extractor_optimizer=self.optimizers.get("feature_extractor"),
                classifier=self.networks["classifier"],
                classifier_optimizer=self.optimizers.get("classifier"),
                images_train=images_train,
                labels_train=self.tensors["labels_train"],
                labels_train_1hot=self.tensors["labels_train_1hot"],
                images_test=images_test,
                labels_test=self.tensors["labels_test"],
                is_train=is_train,
                base_ids=base_ids,
            )

        if not is_train:
            record, self.accuracies = fs_utils.compute_95confidence_intervals(
                record,
                episode=self.biter,
                num_episodes=self.bnumber,
                store_accuracies=self.accuracies,
                metrics=["AccuracyNovel",],
            )

        return record
