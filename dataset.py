import torchvision.datasets as datasets
import numpy as np
from PIL import Image


class TripletDataSet(datasets.ImageFolder):
    """
    This code is a modification of https://github.com/adambielski/siamese-triplet/blob/master/datasets.py#L79
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, image_folder: datasets.ImageFolder, is_train_data: bool = True):
        self.image_folder = image_folder
        self.transform = self.image_folder.transform
        self.loader = self.image_folder.loader
        self.is_train_data = is_train_data

        self.targets = self.image_folder.targets
        self.imgs = self.image_folder.imgs
        self.classes = set(self.image_folder.class_to_idx.values())

        target_array = np.array(self.targets)
        self.label_to_indices = {label: np.where(target_array == label)[0]
                                 for label in self.classes}

        if not self.is_train_data:
            random_state = np.random.RandomState(29)

            triplets = [[anchor_index,
                         random_state.choice(self.label_to_indices[anchor_label]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.classes - set([anchor_label]))
                                                 )
                                             ])
                         ]
                        for anchor_index, anchor_label in enumerate(self.targets)]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.is_train_data:
            anchor, anchor_label = self.imgs[index], self.targets[index]
            positive_index = index
            # Second condition to make sure that it does not get stuck in infinite loop.
            while positive_index == index and len(self.label_to_indices[anchor_label]) > 1:
                positive_index = np.random.choice(self.label_to_indices[anchor_label])
            negative_label = np.random.choice(list(self.classes - set([anchor_label])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            positive_img = self.imgs[positive_index]
            negative_img = self.imgs[negative_index]
        else:
            anchor = self.imgs[self.test_triplets[index][0]]
            positive_img = self.imgs[self.test_triplets[index][1]]
            negative_img = self.imgs[self.test_triplets[index][2]]

        anchor = self.loader(anchor[0])
        positive_img = self.loader(positive_img[0])
        negative_img = self.loader(negative_img[0])
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        return (anchor, positive_img, negative_img), []

    def __len__(self):
        return len(self.image_folder)
