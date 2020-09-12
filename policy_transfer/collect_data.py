import argparse
import pathlib

import numpy as np
import tqdm
import carla
import cv2

from PIL import Image

from .envs.carla_env import CarlaEnv
from .utils.common import visualize_birdview, colorize_segmentation


def save(save_dir, observations, step, debug):
    rgb = observations['rgb']
    birdview = observations['birdview']
    segmentation = observations['segmentation']

    pos = observations['position']
    ori = observations['orientation']
    measurements = np.float32([pos[0], pos[1], pos[2], np.arctan2(ori[1], ori[0])])

    np.save(save_dir / ('measurements_%04d' % step), measurements)
    np.save(save_dir / ('birdview_%04d' % step), birdview)

    Image.fromarray(rgb).save(save_dir / ('image_%04d.png' % step))
    Image.fromarray(segmentation).save(save_dir / ('segmentation_%04d.png' % step))

    if debug:
        cv2.imshow('rgb', cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_BGR2RGB))
        cv2.imshow('birdview', cv2.cvtColor(visualize_birdview(birdview), cv2.COLOR_BGR2RGB))
        cv2.imshow('segmentation', cv2.cvtColor(colorize_segmentation(segmentation), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)


def collect_episode(env, save_dir, episode_length, frame_skip, debug):
    save_dir.mkdir()

    for step in tqdm.tqdm(range(episode_length)):
        spectator = env._world.get_spectator()
        spectator.set_transform(
                carla.Transform(
                    env._player.get_location() + carla.Location(z=75),
                    carla.Rotation(pitch=-90)))

        observations = env.step()

        if step % frame_skip == 0:
            save(save_dir, observations, step // frame_skip, debug)


def main(config):
    np.random.seed(0)

    with CarlaEnv(town='Town06', port=config.port) as env:
        for episode in range(config.episodes):
            env.reset(
                    n_vehicles=np.random.choice([0, 50, 100]),
                    n_pedestrians=0, seed=episode)
            env._player.set_autopilot(True)

            collect_episode(
                    env,
                    config.save_dir / ('%03d' % episode),
                    config.episode_length, config.frame_skip, config.debug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--save_dir', type=pathlib.Path, default='data')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--frame_skip', type=int, default=10)
    parser.add_argument('--debug', action='store_true', default=False)

    main(parser.parse_args())
