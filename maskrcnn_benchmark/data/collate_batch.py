# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))  # data get from Dataset __getitem__() function
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        proposals = transposed_batch[2]
        index = transposed_batch[4]
        img_id = transposed_batch[3]
        return images, targets, proposals, img_id, index
