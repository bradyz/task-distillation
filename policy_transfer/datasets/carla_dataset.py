from pathlib import Path

import numpy as np
import torch
import imgaug.augmenters as iaa

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from .wrapper import Wrap


np.random.seed(0)
torch.manual_seed(0)

# Data has frame skip of 5.
GAP = 4
STEPS = 5

COMMANDS = 6

CROP_SIZE = 192
MAP_SIZE = 320
BIRDVIEW_CHANNELS = 7

MAPPING = np.uint8([
    2,    # unlabeled
    2,    # building
    2,    # fence
    2,    # other
    2,    # ped
    2,    # pole
    0,    # road line
    0,    # road
    2,    # sidewalk
    2,    # vegetation
    1,    # car
    2,    # wall
    2     # traffic sign
    ])


def get_dataset(dataset_dir, batch_size=128, num_workers=4, **kwargs):
    def make_dataset(train_or_val):
        data = list()
        transform = transforms.Compose([
            get_augmenter() if train_or_val == 'train' else lambda x: x,
            transforms.ToTensor()
            ])

        episodes = list((Path(dataset_dir) / train_or_val).glob('*'))
        num_episodes = int(max(1, kwargs.get('dataset_size', 1.0) * len(episodes)))

        for _dataset_dir in episodes[:num_episodes]:
            data.append(CarlaDataset(str(_dataset_dir), transform, **kwargs))

        print('%d frames.' % sum(map(len, data)))

        data = torch.utils.data.ConcatDataset(data)
        data = Wrap(data, batch_size, 1000 if train_or_val == 'train' else 100, num_workers)

        return data

    train = make_dataset('train')
    val = make_dataset('val')

    return train, val


def get_augmenter():
    seq = iaa.Sequential([
        iaa.Sometimes(0.05, iaa.GaussianBlur((0.0, 1.3))),
        iaa.Sometimes(0.05, iaa.AdditiveGaussianNoise(scale=(0.0, 0.05 * 255))),
        iaa.Sometimes(0.05, iaa.Dropout((0.0, 0.1))),
        iaa.Sometimes(0.10, iaa.Add((-0.05 * 255, 0.05 * 255), True)),
        iaa.Sometimes(0.20, iaa.Add((0.25, 2.5), True)),
        iaa.Sometimes(0.05, iaa.contrast.LinearContrast((0.5, 1.5))),
        iaa.Sometimes(0.05, iaa.MultiplySaturation((0.0, 1.0))),
        ])

    return seq.augment_image


def crop_birdview(birdview, dx=0, dy=0):
    x = 238 - CROP_SIZE // 2 + dx
    y = MAP_SIZE // 2 + dy

    birdview = birdview[
            x-CROP_SIZE//2:x+CROP_SIZE//2,
            y-CROP_SIZE//2:y+CROP_SIZE//2]

    return birdview


class CarlaDataset(Dataset):
    def __init__(self, dataset_dir, transform=transforms.ToTensor(), target='map', **kwargs):
        dataset_dir = Path(dataset_dir)

        self.dataset_dir = dataset_dir
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.target = target

        self.frames = list()

        for image_path in sorted(dataset_dir.glob('image*.png')):
            frame = str(image_path.stem).split('_')[-1]

            assert (dataset_dir / ('image_%s.png' % frame)).exists()
            assert (dataset_dir / ('segmentation_%s.png' % frame)).exists()
            assert (dataset_dir / ('birdview_%s.npy' % frame)).exists()
            assert (dataset_dir / ('measurements_%s.npy' % frame)).exists()

            self.frames.append(frame)

    def __len__(self):
        return len(self.frames) - GAP * STEPS

    def __getitem__(self, idx):
        path = self.dataset_dir
        frame = self.frames[idx]

        rgb = np.array(Image.open(str(path / ('image_%s.png' % frame))))
        rgb = self.transform(rgb)
        measurements = np.load(str(path / ('measurements_%s.npy' % frame)))

        if self.target == 'map':
            x, y, z, angle = measurements
            dx, dy = 0, 0

            target = np.load(str(path / ('birdview_%s.npy' % frame)))
            target = crop_birdview(target, dx, dy)
            target = self.to_tensor(target)
            target = target[[0, 5]]
        else:
            target = np.array(Image.open(str(path / ('segmentation_%s.png' % frame))))
            target = MAPPING[target]
            target = torch.LongTensor(target)

        return rgb, target, '%s %s' % (path.stem, frame)


if __name__ == '__main__':
    import sys
    import cv2
    from PIL import ImageDraw
    from ..utils import visualize_birdview
    from ..converter import ConverterTorch

    data = CarlaDataset(sys.argv[1])
    convert = ConverterTorch()

    for i in range(len(data)):
        rgb, birdview, meta = data[i]
        canvas = np.uint8(birdview.detach().cpu().numpy().transpose(1, 2, 0) * 255).copy()
        canvas = visualize_birdview(canvas)
        canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(canvas)

        origin = np.array([birdview.shape[-1] // 2, birdview.shape[-1] - 5])
        offsets = np.array([[0, 0], [0, -10], [-5, -20]])
        points = origin + offsets
        points_cam = convert(torch.FloatTensor(points))
        points_reproject = convert.cam_to_map(points_cam)

        for x, y in points:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 0, 255))

        for x, y in points_reproject:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 0))

        canvas_rgb = np.uint8(rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255).copy()
        canvas_rgb = Image.fromarray(canvas_rgb)
        draw_rgb = ImageDraw.Draw(canvas_rgb)

        for x, y in points_cam:
            draw_rgb.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 0, 255))


        cv2.imshow('map', cv2.cvtColor(np.array(canvas), cv2.COLOR_BGR2RGB))
        cv2.imshow('rgb', cv2.cvtColor(np.array(canvas_rgb), cv2.COLOR_BGR2RGB))
        cv2.waitKey(200)
