from pathlib import Path

import numpy as np
import torch
import torchvision

from scipy import interpolate

from ..models.network import Network
from ..utils.converter import ConverterTorch
from ..utils.common import load_yaml


def subsample(points, steps):
    dists = np.sqrt(((points[1:] - points[:-1]) ** 2).sum(1))
    total_dist = dists.sum()

    result = [points[0]]
    cumulative = 0.0
    index = 0

    for i in range(1, steps):
        while index < len(dists) and cumulative < (i / steps) * total_dist:
            cumulative += dists[index]
            index += 1

        result.append(points[index])

    return np.array(result)


def spline(points, steps):
    t = np.linspace(0.0, 1.0, steps * 10)
    tck, u = interpolate.splprep(points.T, k=2, s=100.0)
    points = np.stack(interpolate.splev(t, tck, der=0), 1)

    return subsample(points, steps)


class Planner(object):
    def __init__(self, path_to_conf_file):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = torchvision.transforms.ToTensor()
        self.converter = ConverterTorch().to(self.device)

        self.target_index = 65
        self.speed_mult = 2.5

        path_to_conf_file = Path(path_to_conf_file)
        config = load_yaml(path_to_conf_file.parent / 'config.yaml')

        self.net = Network(**config['model_args']).to(self.device)
        self.net.load_state_dict(torch.load(path_to_conf_file))
        self.net.eval()

    @torch.no_grad()
    def run_step(self, rgb, viz=None):
        img = self.transform(rgb).to(self.device).unsqueeze(0)

        cam_coords = self.net(img)
        cam_coords[..., 0] = (cam_coords[..., 0] + 1) / 2 * img.shape[-1]
        cam_coords[..., 1] = (cam_coords[..., 1] + 1) / 2 * img.shape[-2]

        map_coords = self.converter.cam_to_map(cam_coords).cpu().numpy().squeeze()
        world_coords = self.converter.cam_to_world(cam_coords).cpu().numpy().squeeze()

        target_speed = np.sqrt(((world_coords[:2] - world_coords[1:3]) ** 2).sum(1).mean())
        target_speed *= self.speed_mult

        curve = spline(map_coords + 1e-8 * np.random.rand(*map_coords.shape), 100)
        target = curve[self.target_index]

        curve_world = spline(world_coords + 1e-8 * np.random.rand(*world_coords.shape), 100)
        target_world = curve_world[self.target_index]

        if viz:
            viz.planner_draw(cam_coords.cpu().numpy().squeeze(), map_coords, curve, target)

        return target_world, target_speed
