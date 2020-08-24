import argparse
import time

from pathlib import Path
from PIL import Image, ImageDraw

import tqdm
import numpy as np
import torch
import torchvision
import wandb

from .datasets import get_dataset, SOURCES
from .models.network import Network
from .utils.common import visualize_birdview


def _log_visuals(birdview, loss, waypoints, _waypoints):
    images = list()

    for i in range(birdview.shape[0]):
        canvas = np.uint8(birdview[i].detach().cpu().numpy().transpose(1, 2, 0) * 255).copy()
        canvas = visualize_birdview(canvas)
        canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(canvas)

        def _dot(i, j, color):
            x = int((i + 1) / 2 * canvas.width)
            y = int((j + 1) / 2 * canvas.height)

            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)

        for x, y in waypoints[i]: _dot(x, y, (0, 0, 255))
        for x, y in _waypoints[i]: _dot(x, y, (255, 0, 0))

        loss_i = loss[i].sum()

        draw.text((5, 10), 'Loss: %.2f' % loss_i)

        images.append((loss_i, torch.ByteTensor(np.uint8(canvas).transpose(2, 0, 1))))

    images.sort(key=lambda x: x[0], reverse=True)

    result = torchvision.utils.make_grid([x[1] for x in images[:32]], nrow=4)
    result = [wandb.Image(result.numpy().transpose(1, 2, 0))]

    return result


def train_or_eval(net, data, optim, is_train, config):
    if is_train:
        desc = 'train'
        net.train()
    else:
        desc = 'val'
        net.eval()

    tick = time.time()
    losses = list()
    iterator = tqdm.tqdm(data, desc=desc, total=len(data), position=1, leave=None)

    for i, (_, mapview, waypoints, _) in enumerate(iterator):
        mapview = mapview.to(config['device'])
        waypoints = waypoints.to(config['device'])

        _waypoints = net(mapview)

        loss = torch.abs(waypoints - _waypoints)
        loss_mean = loss.sum((1, 2)).mean()
        losses.append(loss_mean.item())

        if is_train:
            loss_mean.backward()
            optim.step()
            optim.zero_grad()

            wandb.run.summary['step'] += 1

        metrics = dict()
        metrics['loss'] = loss_mean.item()
        metrics['images_per_second'] = mapview.shape[0] / (time.time() - tick)

        if i % 1000 == 0:
            metrics['images'] = _log_visuals(mapview, loss, waypoints, _waypoints)

        wandb.log(
                {('%s/%s' % (desc, k)): v for k, v in metrics.items()},
                step=wandb.run.summary['step'])

        tick = time.time()

    return np.mean(losses)


def resume_project(net, optim, scheduler, config):
    print('Resumed at epoch %d.' % wandb.run.summary['epoch'])

    net.load_state_dict(torch.load(config['checkpoint_dir'] / 'model_latest.t7'))
    optim.load_state_dict(torch.load(config['checkpoint_dir'] / 'optim_latest.t7'))
    scheduler.load_state_dict(torch.load(config['checkpoint_dir'] / 'scheduler_latest.t7'))


def checkpoint_project(net, optim, scheduler, config):
    torch.save(net.state_dict(), config['checkpoint_dir'] / 'model_latest.t7')
    torch.save(optim.state_dict(), config['checkpoint_dir'] / 'optim_latest.t7')
    torch.save(scheduler.state_dict(), config['checkpoint_dir'] / 'scheduler_latest.t7')


def main(config):
    net = Network(**config['model_args']).to(config['device'])
    data_train, data_val = get_dataset(config['source'])(**config['data_args'])

    optim = torch.optim.Adam(net.parameters(), **config['optimizer_args'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim, milestones=[mult * config['max_epoch'] for mult in [0.5, 0.75]],
            gamma=0.5)

    wandb.init(
            project='task-distillation',
            config=config, id=config['run_name'], resume='auto')
    wandb.save(str(Path(wandb.run.dir) / '*.t7'))

    if wandb.run.resumed:
        resume_project(net, optim, scheduler, config)
    else:
        wandb.run.summary['step'] = 0
        wandb.run.summary['epoch'] = 0
        wandb.run.summary['best_epoch'] = 0

    resume_epoch = max(wandb.run.summary['epoch'], wandb.run.summary['best_epoch'])

    for epoch in tqdm.tqdm(range(resume_epoch+1, config['max_epoch']+1), desc='epoch', position=0):
        wandb.run.summary['epoch'] = epoch

        checkpoint_project(net, optim, scheduler, config)

        loss_train = train_or_eval(net, data_train, optim, True, config)

        with torch.no_grad():
            loss_val = train_or_eval(net, data_val, None, False, config)

        wandb.log({'train/loss_epoch': loss_train, 'val/loss_epoch': loss_val})

        if loss_val < wandb.run.summary.get('best_val_loss', np.inf):
            wandb.run.summary['best_val_loss'] = loss_val
            wandb.run.summary['best_epoch'] = epoch

        if epoch % 10 == 0:
            torch.save(net.state_dict(), Path(wandb.run.dir) / ('model_%03d.t7' % epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--checkpoint_dir', type=Path, default='checkpoints')

    # Model args.
    parser.add_argument('--resnet_model', default='resnet18')
    parser.add_argument('--input_channels', type=int, required=True)
    parser.add_argument('--temperature', type=float, default=1.0)

    # Data args.
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--source', type=str, required=True, choices=SOURCES)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-6)

    parsed = parser.parse_args()

    keys = ['resnet_model', 'lr', 'weight_decay', 'batch_size', 'temperature']
    run_name  = 'stage1' + '_'.join(str(getattr(parsed, x)) for x in keys)

    checkpoint_dir = parsed.checkpoint_dir / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = {
            'run_name': run_name,
            'max_epoch': parsed.max_epoch,
            'checkpoint_dir': checkpoint_dir,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'source': parsed.source,
            'model_args': {
                'temperature': parsed.temperature,
                'resnet_model': parsed.resnet_model,
                'input_channel': parsed.input_channels,
                },
            'data_args': {
                'dataset_dir': parsed.dataset_dir,
                'batch_size': parsed.batch_size,
                },
            'optimizer_args': {
                'lr': parsed.lr,
                'weight_decay': parsed.weight_decay,
                }
            }

    main(config)
