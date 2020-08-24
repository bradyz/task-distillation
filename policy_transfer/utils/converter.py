import numpy as np
import torch


PIXELS_PER_WORLD = 5


class ConverterTorch(torch.nn.Module):
    def __init__(
            self, w=384, h=160, fov=90,
            pixels_per_world=PIXELS_PER_WORLD, map_size=192,
            hack=3, cam_height=1.4):
        super().__init__()

        W = w
        H = h
        FOV = fov
        F = W / (2 * np.tan(FOV * np.pi / 360))

        A = np.array([
            [F, 0, W/2],
            [0, F, H/2],
            [0, 0,   1]
        ])

        self.map_size = map_size
        self.meters_per_pixel = 1 / pixels_per_world
        self.W = W
        self.H = H
        self.F = F
        self.hack = hack
        self.cam_height = cam_height

        self.register_buffer('A', torch.FloatTensor(A))
        self.register_buffer('pos_map', torch.FloatTensor([map_size // 2, map_size]))

    def map_to_world(self, pixel_coords):
        relative_pixel = pixel_coords - self.pos_map
        relative_pixel[..., 1] *= -1

        return relative_pixel * self.meters_per_pixel

    def world_to_cam(self, world_coords):
        world_x = world_coords[..., 0].reshape(-1)
        world_y = world_coords[..., 1].reshape(-1) + self.hack
        world_z = torch.FloatTensor(world_x.shape[0] * [self.cam_height])
        world_z = world_z.to(world_coords.device)

        xyz = torch.stack([world_x, world_z, world_y], -1)

        result = xyz.matmul(self.A.T)
        result = result[:, :2] / result[:, -1].unsqueeze(1)

        result[:, 0] = torch.clamp(result[:, 0], 0, self.W)
        result[:, 1] = torch.clamp(result[:, 1], 0, self.H)

        return result.reshape(*world_coords.shape)

    def forward(self, map_coords):
        world_coords = self.map_to_world(map_coords)
        cam_coords = self.world_to_cam(world_coords)

        return cam_coords

    def cam_to_world(self, points):
        xt = (points[..., 0] - self.A[0, 2]) / self.A[0, 0]
        yt = (points[..., 1] - self.A[1, 2]) / self.A[1, 1]

        world_z = self.cam_height / (yt + 1e-8)
        world_x = world_z * xt

        world_output = torch.stack([world_x, world_z], -1)
        world_output[..., 1] -= 3
        world_output = world_output.squeeze()

        return world_output

    def world_to_map(self, world):
        pixel = world / self.meters_per_pixel
        pixel[..., 1] *= -1

        pixel_coords = pixel + self.pos_map

        return pixel_coords

    def cam_to_map(self, points):
        return self.world_to_map(self.cam_to_world(points))
