from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torchvision import transforms
from PIL import Image

from .wrapper import Wrap


COMMANDS = 5

PIXELS_PER_WORLD = 6
MAP_SIZE = 40               # zoom
AUG_MAP_SIZE = 192          # resize
STEPS = 5
GAP = 4


def get_dataset(dataset_dir, batch_size=128, num_workers=2, verbose=False, **kwargs):

    def make_dataset(is_train):
        data = list()
        train_or_val = 'train' if is_train else 'val'
        episodes = Path(dataset_dir) / train_or_val

        for episode_dir in episodes.iterdir():
            data.append(SuperTuxKartDataset(episode_dir, **kwargs))

        data = torch.utils.data.ConcatDataset(data)

        print(train_or_val, len(data))

        return Wrap(data, batch_size, 1000 if is_train else 100, num_workers)

    train_dataset = make_dataset(True)
    test_dataset = make_dataset(False)

    return train_dataset, test_dataset


class SuperTuxKartDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, gap=GAP, steps=STEPS, target='map'):
        self.gap = gap
        self.steps = steps

        self.dataset_dir = Path(dataset_dir)

        target_map = {
                'map': 'birdview',
                'segmentation': 'segmentation'
                }

        self.target = target_map[target]

        self.measurements = pd.read_csv(self.dataset_dir / 'episode.csv')
        self.imgs = list(sorted(self.dataset_dir.glob('rgb_*.png')))
        self.targets = list(sorted(self.dataset_dir.glob('%s_*' % self.target)))
        self.waypoints = self._generate_waypoints()

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.waypoints)

    def __getitem__(self, idx):
        rgb = self.transform(Image.open(self.imgs[idx]))
        target = torch.FloatTensor(np.load(self.targets[idx]))[:1]
        # target = self.transform(np.expand_dims(np.load(self.targets[idx]), -1))
        waypoints = torch.FloatTensor(self.waypoints[idx].copy())

        return rgb, target, waypoints, ('%s %s' % (self.dataset_dir, idx))

    def _get_points(self, i):
        window = self.measurements[i:i+self.gap*(self.steps+1):self.gap]
        xy = np.stack((window['x'], window['y']), -1)

        direction = xy[1:] - xy[0]
        angle = window['rotation'].iloc[0] - np.pi / 2

        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])

        points = direction.dot(R) * PIXELS_PER_WORLD

        # pixel coords.
        points[:, 0] = AUG_MAP_SIZE // 2 + points[:, 0]
        points[:, 1] = AUG_MAP_SIZE - points[:, 1] + 10

        # normalize
        points[:, 0] = (points[:, 0] / AUG_MAP_SIZE) * 2 - 1
        points[:, 1] = (points[:, 1] / AUG_MAP_SIZE) * 2 - 1

        return points

    def _generate_waypoints(self):
        waypoints = list()

        for i in range(len(self.measurements) - (self.steps * self.gap + self.gap)):
            waypoints.append(self._get_points(i))

        return np.float32(waypoints)


if __name__ == '__main__':
    import sys
    import cv2

    from PIL import ImageDraw
    from .highway_dataset import ConverterTorch

    def _dot(_canvas, _draw, x, y, color, is_normalized=True):
        if is_normalized:
            x = int((x + 1) / 2 * _canvas.width)
            y = int((y + 1) / 2 * _canvas.height)

        _draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)

    data = SuperTuxKartDataset(sys.argv[1], 5, target='segmentation')
    converter = ConverterTorch()

    for i in range(len(data)):
        rgb, mapview, waypoints, meta = data[i]

        canvas = Image.fromarray(255 * mapview.squeeze().byte().numpy()).convert('RGB')
        draw = ImageDraw.Draw(canvas)

        waypoints_unnormalized = torch.FloatTensor(waypoints)
        waypoints_unnormalized[..., 0] = (waypoints_unnormalized[..., 0] + 1) * 192 / 2
        waypoints_unnormalized[..., 1] = (waypoints_unnormalized[..., 1] + 1) * 192 / 2

        waypoints_rgb = converter(waypoints_unnormalized).squeeze()

        # for x, y in waypoints.squeeze():
            # _dot(canvas, draw, x, y, (0, 0, 255), is_normalized=True)

        for x, y in waypoints_rgb:
            _dot(canvas, draw, x, y, (0, 0, 255), False)

        cv2.imshow('mapview', cv2.cvtColor(np.array(canvas), cv2.COLOR_BGR2RGB))

        # canvas = Image.fromarray(np.uint8(rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255))
        # draw = ImageDraw.Draw(canvas)

        # waypoints_unnormalized = waypoints
        # waypoints_unnormalized[..., 0] = (waypoints[..., 0] + 1) * mapview.shape[-1] / 2
        # waypoints_unnormalized[..., 1] = (waypoints[..., 1] + 1) * mapview.shape[-2] / 2

        # for x, y in converter(waypoints_unnormalized).squeeze():
            # _dot(canvas, draw, x, y, (0, 0, 255), False)

        # cv2.imshow('rgb', cv2.cvtColor(np.array(canvas), cv2.COLOR_BGR2RGB))

        cv2.waitKey(10)
