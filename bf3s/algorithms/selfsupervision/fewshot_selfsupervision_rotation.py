import torch

import bf3s.algorithms.algorithm as algorithm
import bf3s.algorithms.fewshot.utils as fs_utils
import bf3s.algorithms.selfsupervision.rotation_utils as rot_utils
import bf3s.utils as utils


class FewShotRotationSelfSupervision(algorithm.Algorithm):
    """Trains a few-shot model with the auxiliary rotation prediction task."""

    def __init__(self, opt, _run=None, _log=None):
        super().__init__(opt, _run, _log)

        self.keep_best_model_metric_name = "AccuracyNovel"
        self.auxiliary_rotation_task_coef = opt["auxiliary_rotation_task_coef"]
        self.rotation_invariant_classifier = opt["rotation_invariant_classifier"]
        self.random_rotation = opt["random_rotation"]
        self.semi_supervised = opt["semi_supervised"] if ("semi_supervised" in opt) else False
        feature_name = opt["feature_name"] if ("feature_name" in opt) else None
        if feature_name:
            assert isinstance(feature_name, (list, tuple))
            assert len(feature_name) == 1
        self.feature_name = feature_name
        self.accuracies = {}

    def allocate_tensors(self):
        self.tensors = {
            "images_train": torch.FloatTensor(),
            "labels_train": torch.LongTensor(),
            "labels_train_1hot": torch.FloatTensor(),
            "images_test": torch.FloatTensor(),
            "labels_test": torch.LongTensor(),
            "Kids": torch.LongTensor(),
            "images_unlabeled": torch.FloatTensor(),
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
            assert len(batch[1]) == 1
            assert self.semi_supervised is True
            images_test, labels_test, K, num_base_per_episode = batch[0]
            (images_unlabeled,) = batch[1]
            self.num_base = num_base_per_episode[0].item()
            self.tensors["images_test"].resize_(images_test.size()).copy_(images_test)
            self.tensors["labels_test"].resize_(labels_test.size()).copy_(labels_test)
            self.tensors["Kids"].resize_(K.size()).copy_(K)
            self.tensors["images_unlabeled"].resize_(images_unlabeled.size()).copy_(
                images_unlabeled
            )
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
        elif len(batch) == 4:
            train_test_stage = "classification"
            images_test, labels_test, K, num_base_per_episode = batch
            self.num_base = num_base_per_episode[0].item()
            self.tensors["images_test"].resize_(images_test.size()).copy_(images_test)
            self.tensors["labels_test"].resize_(labels_test.size()).copy_(labels_test)
            self.tensors["Kids"].resize_(K.size()).copy_(K)

        return train_test_stage

    def train_step(self, batch):
        return self.process_batch(batch, is_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, is_train=False)

    def process_batch(self, batch, is_train):
        process_type = self.set_tensors(batch)
        auxiliary_rotation_task = is_train and (self.auxiliary_rotation_task_coef > 0.0)
        if process_type == "fewshot":
            record = self.process_batch_fewshot_classification_task(
                auxiliary_rotation_task, is_train
            )
        elif process_type == "classification":
            record = self.process_batch_base_class_classification_task(
                auxiliary_rotation_task, is_train
            )
        else:
            raise ValueError(f"Unexpected process type {process_type}")

        return record

    def process_batch_base_class_classification_task(self, auxiliary_rotation_task, is_train):

        images = self.tensors["images_test"]
        labels = self.tensors["labels_test"]
        Kids = self.tensors["Kids"]
        assert images.dim() == 5 and labels.dim() == 2
        images = utils.convert_from_5d_to_4d(images)
        labels = labels.view(-1)

        if self.semi_supervised and is_train:
            images_unlabeled = self.tensors["images_unlabeled"]
            assert images_unlabeled.dim() == 4
            assert auxiliary_rotation_task is True
        else:
            images_unlabeled = None

        if auxiliary_rotation_task:
            record = rot_utils.object_classification_with_rotation_selfsupervision(
                feature_extractor=self.networks["feature_extractor"],
                feature_extractor_optimizer=self.optimizers["feature_extractor"],
                classifier=self.networks["classifier"],
                classifier_optimizer=self.optimizers["classifier"],
                classifier_rot=self.networks["classifier_aux"],
                classifier_rot_optimizer=self.optimizers["classifier_aux"],
                images=images,
                labels=labels,
                is_train=is_train,
                alpha=self.auxiliary_rotation_task_coef,
                random_rotation=self.random_rotation,
                rotation_invariant_classifier=self.rotation_invariant_classifier,
                base_ids=Kids[:, : self.num_base].contiguous(),
                feature_name=self.feature_name,
                images_unlabeled=images_unlabeled,
            )
        else:
            record = rot_utils.object_classification_rotation_invariant(
                feature_extractor=self.networks["feature_extractor"],
                feature_extractor_optimizer=self.optimizers["feature_extractor"],
                classifier=self.networks["classifier"],
                classifier_optimizer=self.optimizers["classifier"],
                images=images,
                labels=labels,
                is_train=is_train,
                rotation_invariant_classifier=self.rotation_invariant_classifier,
                random_rotation=self.random_rotation,
                base_ids=Kids[:, : self.num_base].contiguous(),
                feature_name=self.feature_name,
            )

        return record

    def process_batch_fewshot_classification_task(self, auxiliary_rotation_task, is_train):

        Kids = self.tensors["Kids"]
        base_ids = None if (self.num_base == 0) else Kids[:, : self.num_base].contiguous()

        if auxiliary_rotation_task:
            if self.rotation_invariant_classifier:
                raise ValueError("Not supported option.")
            if self.random_rotation:
                raise ValueError("Not supported option.")
            if self.semi_supervised:
                raise ValueError("Not supported option.")

            record = rot_utils.fewshot_classification_with_rotation_selfsupervision(
                feature_extractor=self.networks["feature_extractor"],
                feature_extractor_optimizer=self.optimizers.get("feature_extractor"),
                classifier=self.networks["classifier"],
                classifier_optimizer=self.optimizers.get("classifier"),
                classifier_rot=self.networks["classifier_aux"],
                classifier_rot_optimizer=self.optimizers.get("classifier_aux"),
                images_train=self.tensors["images_train"],
                labels_train=self.tensors["labels_train"],
                labels_train_1hot=self.tensors["labels_train_1hot"],
                images_test=self.tensors["images_test"],
                labels_test=self.tensors["labels_test"],
                is_train=is_train,
                alpha=self.auxiliary_rotation_task_coef,
                base_ids=base_ids,
                feature_name=self.feature_name,
            )
        else:
            record = fs_utils.fewshot_classification(
                feature_extractor=self.networks["feature_extractor"],
                feature_extractor_optimizer=self.optimizers.get("feature_extractor"),
                classifier=self.networks["classifier"],
                classifier_optimizer=self.optimizers.get("classifier"),
                images_train=self.tensors["images_train"],
                labels_train=self.tensors["labels_train"],
                labels_train_1hot=self.tensors["labels_train_1hot"],
                images_test=self.tensors["images_test"],
                labels_test=self.tensors["labels_test"],
                is_train=is_train,
                base_ids=base_ids,
                feature_name=self.feature_name,
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
