from pathlib import Path

import numpy as np
import yaml


CROP_SIZE = 192
MAP_SIZE = 320
PIXELS_PER_METER = 5
BACKGROUND = [0, 47, 0]
COLORS = [
        (102, 102, 102),
        (253, 253, 17),
        (204, 6, 5),
        (250, 210, 1),
        (39, 232, 51),
        (0, 0, 142),
        (220, 20, 60)
        ]


def visualize_birdview(birdview):
    """
    0 road
    1 lane
    2 red light
    3 yellow light
    4 green light
    5 vehicle
    6 pedestrian
    """
    BACKGROUND = [0, 0, 0]
    COLORS = [
            (128,  64, 128),
            (  0,   0, 142),
            (204, 6, 5),
            (250, 210, 1),
            (39, 232, 51),
            (0, 0, 142),
            (220, 20, 60)
            ]

    h, w = birdview.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[...] = BACKGROUND

    for i in range(min(len(COLORS), birdview.shape[2])):
        canvas[birdview[:, :, i] > 0] = COLORS[i]

    return canvas


def colorize_segmentation(segmentation):
    colors = np.uint8([
            (  0,   0,   0),    # unlabeled
            ( 70,  70,  70),    # building
            (190, 153, 153),    # fence
            (250, 170, 160),    # other
            (220,  20,  60),    # ped
            (153, 153, 153),    # pole
            (157, 234,  50),    # road line
            (128,  64, 128),    # road
            (244,  35, 232),    # sidewalk
            (107, 142,  35),    # vegetation
            (  0,   0, 142),    # car
            (102, 102, 156),    # wall
            (220, 220,   0)     # traffic sign
            ])

    return colors[segmentation]


def load_yaml(yml_path):
    return {k: v['value'] for k, v in yaml.load(Path(yml_path).read_text()).items() if isinstance(v, dict)}
