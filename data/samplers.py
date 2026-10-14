"""GSV place-based pair sampling, separate from image and graph loading."""

import torch
from torch.utils.data import Sampler


def validate_pair_batch_settings(batch_size, images_per_place):
    """B pairs use P places x K unique images, with 2*B=P*K and 50/50 labels."""
    if type(images_per_place) is not int or images_per_place < 4 or images_per_place % 4:
        raise ValueError("GSV images_per_place must be a positive multiple of 4")
    if (type(batch_size) is not int or batch_size < images_per_place
            or 2 * batch_size % images_per_place):
        raise ValueError("GSV batch_size counts pairs: require 2*batch_size=P*K, P>=2, K=images_per_place")
    return 2 * batch_size // images_per_place


class PairBatchSampler(Sampler):
    """B pairs, half same-place and half different-place; each image used once.

    Sample K distinct views per place. Half form within-place positives;
    remaining views form negatives across a cycle of distinct places.
    set_epoch(epoch) resamples views, pair partners and A/B orientation.
    Partial batches need at least two places; a singleton tail is dropped.
    """

    def __init__(self, dataset, batch_size=8, images_per_place=4,
                 shuffle=True, drop_last=True, seed=42):
        self.places_per_batch = validate_pair_batch_settings(batch_size, images_per_place)
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.batch_size = batch_size
        self.images_per_place = images_per_place
        self.shuffle, self.drop_last, self.seed = shuffle, drop_last, seed
        self.epoch = 0
        self.groups = [indices for indices in dataset.place_to_indices.values()
                       if len(indices) >= images_per_place]
        self.excluded_places = len(dataset.place_to_indices) - len(self.groups)
        self.eligible_images = sum(map(len, self.groups))
        if len(self.groups) < self.places_per_batch:
            raise ValueError("Not enough eligible places for one GSV pair batch")

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def __len__(self):
        full, remaining = divmod(len(self.groups), self.places_per_batch)
        return full + int(not self.drop_last and remaining >= 2)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.groups), generator=generator).tolist() if self.shuffle else list(range(len(self.groups)))
        quarter = self.images_per_place // 4
        for start in range(0, len(order), self.places_per_batch):
            places = order[start:start + self.places_per_batch]
            if len(places) < 2 or self.drop_last and len(places) < self.places_per_batch:
                break
            selected = []
            for place in places:
                indices = self.groups[place]
                views = torch.randperm(len(indices), generator=generator)[:self.images_per_place].tolist()
                selected.append([indices[view] for view in views])
            pairs = []
            for position, views in enumerate(selected):
                pairs.extend((views[i], views[i + 1]) for i in range(0, 2 * quarter, 2))
                partner = selected[(position + 1) % len(selected)]
                pairs.extend((views[2 * quarter + i], partner[3 * quarter + i]) for i in range(quarter))
            # Random branch direction and pair order preserve image/graph keys.
            pairs = [pair[::-1] if torch.randint(2, (), generator=generator).item() else pair for pair in pairs]
            if self.shuffle:
                pairs = [pairs[i] for i in torch.randperm(len(pairs), generator=generator).tolist()]
            yield pairs
