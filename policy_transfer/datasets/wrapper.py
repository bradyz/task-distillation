from itertools import repeat

import torch


def repeater(loader):
    for loader in repeat(loader):
        for data in loader:
            yield data


def dataloader(data, batch_size, num_workers):
    return torch.utils.data.DataLoader(
            data, batch_size=batch_size, num_workers=num_workers,
            shuffle=True, drop_last=True, pin_memory=True)


def infinite_dataloader(data, batch_size, num_workers):
    return repeater(dataloader(data, batch_size, num_workers))


class Wrap(object):
    def __init__(self, data, batch_size, samples, num_workers):
        self.data = infinite_dataloader(data, batch_size, num_workers)
        self.samples = samples
        self.count = 0

    def __iter__(self):
        for i in range(self.samples):
            yield next(self.data)

    def __len__(self):
        return self.samples
