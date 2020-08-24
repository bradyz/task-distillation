from pathlib import Path

import numpy as np
import torch
import cv2
import pandas as pd

from torchvision import transforms
from PIL import Image

from .wrapper import Wrap


PIXELS_PER_WORLD = 5
AUG_MAP_SIZE = 192
STEPS = 5
GAP = 3


def get_dataset(dataset_dir, batch_size=128, num_workers=4, **kwargs):
    def make_dataset(is_train):
        data = list()
        train_or_val = 'train' if is_train else 'val'

        for episode_dir in (Path(dataset_dir) / train_or_val).iterdir():
            data.append(VizDoomWaypointDataset(episode_dir, **kwargs))

        data = torch.utils.data.ConcatDataset(data)

        print('%s: %d' % (train_or_val, len(data)))

        return Wrap(data, batch_size, 1000 if is_train else 100, num_workers)

    train_dataset = make_dataset(True)
    test_dataset = make_dataset(False)

    return train_dataset, test_dataset


class VizDoomWaypointDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, target='map'):
        self.steps = STEPS
        self.gap = GAP

        if not isinstance(dataset_dir, Path):
            dataset_dir = Path(dataset_dir)

        self.dataset_dir = dataset_dir
        self.target = target

        self.measurements = pd.read_csv(dataset_dir / 'episode.csv')
        self.imgs = list(sorted(dataset_dir.glob('rgb_*.png')))
        self.maps = list(sorted(dataset_dir.glob('birdview_*.png')))
        self.segs = list(sorted(dataset_dir.glob('segmentation_*.npy')))
        self.waypoints = self._generate_waypoints()

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.waypoints)

    def __getitem__(self, idx):
        rgb = self.transform(Image.open(self.imgs[idx]))
        waypoints = torch.FloatTensor(self.waypoints[idx].copy())

        if self.target == 'map':
            target = self.transform(Image.open(self.maps[idx])).round()[[0, 1]]
        else:
            target = torch.Tensor(np.load(self.segs[idx]))

        return rgb, target, waypoints, ('%s %s' % (self.dataset_dir, idx))

    def _get_points(self, i):
        window = self.measurements[i:i+self.steps*self.gap+self.gap:self.gap]
        xy = np.stack((window['x'], window['y']), -1)

        angle = np.deg2rad(self.measurements.iloc[i]['angle'] - 90)
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])

        points = (xy[1:] - xy[0]).dot(R) / 4

        # pixel coords.
        points[:, 0] = AUG_MAP_SIZE // 2 + points[:, 0]
        points[:, 1] = AUG_MAP_SIZE - points[:, 1]

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

    from PIL import ImageDraw

    from .common import visualize_birdview
    from .carla_dataset import ConverterTorch

    data = VizDoomWaypointDataset(sys.argv[1], target='map')

    def _dot(_canvas, _draw, x, y, color, is_normalized=True):
        if is_normalized:
            x = int((x + 1) / 2 * _canvas.width)
            y = int((y + 1) / 2 * _canvas.height)

        _draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)

    for i in range(0, len(data)):
        rgb, mapview, waypoints, _ = data[i]

        canvas_map = np.uint8(mapview.detach().cpu().numpy().transpose(1, 2, 0) * 255).copy()
        canvas_map = visualize_birdview(canvas_map)
        canvas_map = Image.fromarray(canvas_map)
        draw_map = ImageDraw.Draw(canvas_map)

        for x, y in waypoints.squeeze():
            _dot(canvas_map, draw_map, x, y, (0, 0, 255), is_normalized=True)

        converter = ConverterTorch()
        canvas = Image.fromarray(np.uint8(rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255))
        draw = ImageDraw.Draw(canvas)

        waypoints_unnormalized = torch.FloatTensor(waypoints)
        waypoints_unnormalized[..., 0] = (waypoints_unnormalized[..., 0] + 1) * 192 / 2
        waypoints_unnormalized[..., 1] = (waypoints_unnormalized[..., 1] + 1) * 192 / 2

        waypoints_rgb = converter(waypoints_unnormalized).squeeze()

        for x, y in waypoints_rgb:
            _dot(canvas, draw, x, y, (0, 0, 255), False)

        for x, y in converter.cam_to_map(waypoints_rgb):
            _dot(canvas_map, draw_map, x, y, (0, 255, 0), False)

        cv2.imshow('mapview', cv2.cvtColor(np.array(canvas_map), cv2.COLOR_BGR2RGB))
        cv2.imshow('rgb', cv2.cvtColor(np.array(canvas), cv2.COLOR_BGR2RGB))

        cv2.waitKey(0)
