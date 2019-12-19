import torch

import bf3s.algorithms.algorithm as algorithm
import bf3s.algorithms.classification.utils as cls_utils
import bf3s.algorithms.fewshot.utils as fs_utils
import bf3s.utils as utils


class FewShot(algorithm.Algorithm):
    """Trains a few-shot model."""

    def __init__(self, opt, _run=None, _log=None):
        super().__init__(opt, _run, _log)
        self.keep_best_model_metric_name = "AccuracyNovel"
        self.nKbase = torch.LongTensor()

        self.all_base_cats = opt["all_base_cats"] if ("all_base_cats" in opt) else False

        self.accuracies = {}

    def allocate_tensors(self):
        self.tensors = {
            "images_train": torch.FloatTensor(),
            "labels_train": torch.LongTensor(),
            "labels_train_1hot": torch.FloatTensor(),
            "images_test": torch.FloatTensor(),
            "labels_test": torch.LongTensor(),
            "Kids": torch.LongTensor(),
        }

    def set_tensors(self, batch):
        self.nKbase = self.dloader.nKbase
        self.nKnovel = self.dloader.nKnovel

        if self.nKnovel > 0:
            train_test_stage = "fewshot"
            assert len(batch) == 6
            images_train, labels_train, images_test, labels_test, K, nKbase = batch
            self.nKbase = nKbase[0].item()
            self.tensors["images_train"].resize_(images_train.size()).copy_(images_train)
            self.tensors["labels_train"].resize_(labels_train.size()).copy_(labels_train)
            labels_train = self.tensors["labels_train"]

            nKnovel = 1 + labels_train.max().item() - self.nKbase

            labels_train_1hot_size = list(labels_train.size()) + [
                nKnovel,
            ]
            labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
            self.tensors["labels_train_1hot"].resize_(labels_train_1hot_size).fill_(
                0
            ).scatter_(
                len(labels_train_1hot_size) - 1, labels_train_unsqueeze - self.nKbase, 1
            )
            self.tensors["images_test"].resize_(images_test.size()).copy_(images_test)
            self.tensors["labels_test"].resize_(labels_test.size()).copy_(labels_test)
            self.tensors["Kids"].resize_(K.size()).copy_(K)
        else:
            train_test_stage = "base_classification"
            assert len(batch) == 4
            images_test, labels_test, K, nKbase = batch
            self.nKbase = nKbase.squeeze().item()
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
        if process_type == "fewshot":
            return self.process_batch_fewshot_classification_task(is_train)
        elif process_type == "base_classification":
            return self.process_batch_base_class_classification_task(is_train)
        else:
            raise ValueError(f"Unexpected process type {process_type}")

    def process_batch_base_class_classification_task(self, is_train):
        images = self.tensors["images_test"]
        labels = self.tensors["labels_test"]
        Kids = self.tensors["Kids"]
        base_ids = None if (self.nKbase == 0) else Kids[:, : self.nKbase].contiguous()

        assert images.dim() == 5 and labels.dim() == 2
        images = utils.convert_from_5d_to_4d(images)
        labels = labels.view(-1)

        if self.optimizers.get("feature_extractor") is None:
            self.networks["feature_extractor"].eval()

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
        nKbase = self.nKbase
        if is_train and self.all_base_cats:
            assert nKbase == 0
            base_ids = Kids
        else:
            base_ids = None if (self.nKbase == 0) else Kids[:, :nKbase].contiguous()

        if self.optimizers.get("feature_extractor") is None:
            self.networks["feature_extractor"].eval()

        record = fs_utils.fewshot_classification(
            feature_extractor=self.networks["feature_extractor"],
            feature_extractor_optimizer=self.optimizers.get("feature_extractor"),
            classifier=self.networks["classifier"],
            classifier_optimizer=self.optimizers["classifier"],
            images_train=self.tensors["images_train"],
            labels_train=self.tensors["labels_train"],
            labels_train_1hot=self.tensors["labels_train_1hot"],
            images_test=self.tensors["images_test"],
            labels_test=self.tensors["labels_test"],
            is_train=is_train,
            base_ids=base_ids,
            classification_coef=self.classification_loss_coef,
        )

        if not is_train:
            metrics = [
                "AccuracyNovel",
            ]
            if "AccuracyBoth" in record:
                metrics.append("AccuracyBoth")
            record, self.accuracies = fs_utils.compute_95confidence_intervals(
                record,
                episode=self.biter,
                num_episodes=self.bnumber,
                store_accuracies=self.accuracies,
                metrics=metrics,
            )

        return record
