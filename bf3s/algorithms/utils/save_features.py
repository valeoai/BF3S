import h5py
import torch
from tqdm import tqdm

import bf3s.algorithms.algorithm as algorithm
import bf3s.algorithms.classification.utils as cls_utils
import bf3s.architectures.tools as tools


class SaveFeatures(algorithm.Algorithm):
    """Extracts features from a dataset."""

    def __init__(self, opt, _run=None, _log=None):
        super().__init__(opt, _run, _log)

    def allocate_tensors(self):
        self.tensors = {"images": torch.FloatTensor(), "labels": torch.LongTensor()}

    def set_tensors(self, batch):
        assert len(batch) == 2
        images, labels = batch
        self.tensors["images"].resize_(images.size()).copy_(images)
        self.tensors["labels"].resize_(labels.size()).copy_(labels)

    def save_features(self, dataloader, filename, feature_name=None, global_pooling=True):
        """Saves features and labels for each image in the dataloader.

        This routines uses the trained feature model (i.e.,
        self.networks['feature_extractor']) in order to extract a feature for each
        image in the dataloader. The extracted features along with the labels
        of the images that they come from are saved in a h5py file.

        Args:
            dataloader: A dataloader that feeds images and labels.
            filename: The file name where the features and the labels of each
                images in the dataloader are saved.
            feature_name:
        """

        if isinstance(feature_name, (list, tuple)):
            assert len(feature_name) == 1

        feature_extractor = self.networks["feature_extractor"]
        feature_extractor.eval()

        self.dloader = dataloader
        dataloader_iterator = dataloader.get_iterator()

        self.logger.info(f"Destination filename for features: {filename}")

        data_file = h5py.File(filename, "w")
        max_count = len(dataloader_iterator) * dataloader_iterator.batch_size
        all_labels = data_file.create_dataset("all_labels", (max_count,), dtype="i")
        all_features = None

        count = 0
        for i, batch in enumerate(tqdm(dataloader_iterator)):
            with torch.no_grad():
                self.set_tensors(batch)
                images = self.tensors["images"].detach()
                labels = self.tensors["labels"].detach()
                assert images.dim() == 4
                assert labels.dim() == 1

                features = cls_utils.extract_features(
                    feature_extractor, images, feature_name=feature_name
                )

                if global_pooling and features.dim() == 4:
                    features = tools.global_pooling(features, pool_type="avg")
                features = features.view(features.size(0), -1)
                assert features.dim() == 2

                if all_features is None:
                    self.logger.info("Image size: {}".format(images.size()))
                    self.logger.info("Feature size: {}".format(features.size()))
                    self.logger.info(f"Max_count: {max_count}")
                    all_features = data_file.create_dataset(
                        "all_features", (max_count, features.size(1)), dtype="f"
                    )
                    self.logger.info("Number of feature channels: {}".format(features.size(1)))

                all_features[count : (count + features.size(0)), :] = features.cpu().numpy()
                all_labels[count : (count + features.size(0))] = labels.cpu().numpy()
                count = count + features.size(0)

        self.logger.info(f"Number of processed primages: {count}")

        count_var = data_file.create_dataset("count", (1,), dtype="i")
        count_var[0] = count
        data_file.close()
