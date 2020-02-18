import numpy as np
import torch
import torchvision.datasets as datasets


class TripletDataSet(datasets.ImageFolder):
    """
    This code is a modification of https://github.com/adambielski/siamese-triplet/blob/master/datasets.py#L79
    Unlike original code, we use the fixed bert tensors for the examples.
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing

    """

    def __init__(self, image_folder: datasets.ImageFolder, bert_tensors: torch.Tensor, num_neg_samples: int = 30,
                 is_train_data: bool = True):
        self.image_folder = image_folder
        self.transform = self.image_folder.transform
        self.loader = self.image_folder.loader
        self.is_train_data = is_train_data
        self.bert_tensors = bert_tensors
        self.num_neg_samples = num_neg_samples

        self.targets = self.image_folder.targets
        self.imgs = self.image_folder.imgs
        self.classes = set(self.image_folder.class_to_idx.values())
        assert len(self.classes) == bert_tensors.size(0)

        target_array = np.array(self.targets)
        self.label_to_indices = {label: np.where(target_array == label)[0]
                                 for label in self.classes}

        if not self.is_train_data:
            self.test_triplets = [np.random.choice(list(self.classes - set([anchor_label])), self.num_neg_samples) for anchor_label in self.targets]

    def __getitem__(self, index):
        anchor, anchor_label = self.imgs[index], self.targets[index]
        if self.is_train_data:
            negative_label = np.random.choice(list(self.classes - set([anchor_label])), self.num_neg_samples)
        else:
            negative_label = self.test_triplets[index]

        anchor = self.loader(anchor[0])

        positive_bert_embedding = self.bert_tensors[anchor_label]
        negative_bert_embedding = self.bert_tensors[negative_label]
        if self.transform is not None:
            anchor = self.transform(anchor)

        return (anchor, positive_bert_embedding, negative_bert_embedding), []

    def __len__(self):
        return len(self.image_folder)
