import numpy as np

from PIL import Image, ImageDraw

from .planner import Planner
from .controller import Controller
from ..utils.common import visualize_birdview
from ..datasets.carla_dataset import crop_birdview


class Visualizer(object):
    def __init__(self, rgb, birdview):
        self.canvas = Image.fromarray(rgb)
        self.draw = ImageDraw.Draw(self.canvas)

        self.canvas_birdview = Image.fromarray(crop_birdview(visualize_birdview(birdview)))
        self.draw_birdview = ImageDraw.Draw(self.canvas_birdview)

        self.step = 0

    def planner_draw(self, cam_coords, map_coords, curve, target):
        for x, y in cam_coords.squeeze():
            self.draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 0))

        for x, y in curve:
            self.draw_birdview.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(0, 0, 255))

        for x, y in map_coords:
            self.draw_birdview.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 0))

        self.draw_birdview.ellipse(
                (target[0] - 2, target[1] - 2, target[0] + 2, target[1] + 2),
                fill=(255, 255, 255))

    def controller_draw(self, speed, target_speed, control):
        control_text = [control.steer, control.throttle, control.brake]
        control_text = ' '.join(str('%.2f' % x).rjust(5, ' ') for x in control_text)

        self.draw.text((5, 10), 'Control: %s' % control_text)
        self.draw.text((5, 20), 'Current Speed: %.2f' % speed)
        self.draw.text((5, 30), 'Target Speed: %.2f' % target_speed)

    def show(self, step):
        self.draw.text((5, 40), 'Seconds: %.2f' % (step / 20))
        self.canvas_birdview.thumbnail(self.canvas.size)

        return np.hstack([np.uint8(self.canvas), np.uint8(self.canvas_birdview)])


class ImageAgent(object):
    def __init__(self, pid, path_to_conf_file):
        self.planner = Planner(path_to_conf_file)
        self.controller = Controller(pid)

        self.step = -1
        self.debug_image = None

    def run_step(self, observations):
        self.step += 1

        rgb = observations['rgb'].copy()
        speed = np.linalg.norm(observations['velocity'])

        viz = Visualizer(rgb, observations['birdview'])

        target, target_speed = self.planner.run_step(rgb, viz)
        control = self.controller.run_step(target, target_speed, speed, viz)

        self.debug_image = viz.show(self.step)

        return control

    def reset(self):
        self.step = -1
        self.debug_image = None
