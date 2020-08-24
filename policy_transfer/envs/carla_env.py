import collections
import queue
import weakref
import time
import random

import numpy as np
import carla

from .map_utils import Wrapper as map_utils


MAX_SPEED_MS = 10
SPAWNS = [
        304,303,302,301,94,93,92,91,90,89,88,87,
        352,351,350,60,61,62,63,313,314,315]


PRESET_WEATHERS = {
    1: carla.WeatherParameters.ClearNoon,
    2: carla.WeatherParameters.CloudyNoon,
    3: carla.WeatherParameters.WetNoon,
    4: carla.WeatherParameters.WetCloudyNoon,
    5: carla.WeatherParameters.MidRainyNoon,
    6: carla.WeatherParameters.HardRainNoon,
    7: carla.WeatherParameters.SoftRainNoon,
    8: carla.WeatherParameters.ClearSunset,
    9: carla.WeatherParameters.CloudySunset,
    10: carla.WeatherParameters.WetSunset,
    11: carla.WeatherParameters.WetCloudySunset,
    12: carla.WeatherParameters.MidRainSunset,
    13: carla.WeatherParameters.HardRainSunset,
    14: carla.WeatherParameters.SoftRainSunset,
}

WEATHERS = list(PRESET_WEATHERS.values())
VEHICLE_NAME = '*mustang*'
COLLISION_THRESHOLD = 10


def _carla_img_to_numpy(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]

    return array


def set_sync_mode(client, sync):
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 0.1

    world.apply_settings(settings)


class CarlaEnv(object):
    def __init__(self, town='Town01', port=2000, **kwargs):
        self._client = carla.Client('localhost', port)
        self._client.set_timeout(30.0)

        set_sync_mode(self._client, False)

        self._town_name = town
        self._world = self._client.load_world(town)
        self._map = self._world.get_map()

        self._blueprints = self._world.get_blueprint_library()

        self._tick = 0
        self._player = None

        # vehicle, sensor
        self._actor_dict = collections.defaultdict(list)

        self.collided = False
        self._collided_frame_number = -1

        self._rgb_queue = None
        self._seg_queue = None

    def _spawn_vehicles(self, n_vehicles, n_retries=10):
        vehicle_blueprints = self._blueprints.filter('vehicle.*')
        vehicle_blueprints = [x for x in vehicle_blueprints if int(x.get_attribute('number_of_wheels')) == 4]

        spawn_points = self._map.get_spawn_points()

        for index, transform in enumerate(spawn_points):
            self._world.debug.draw_string(
                    transform.location + carla.Location(z=1),
                    str(index), draw_shadow=False, color=carla.Color(255, 255, 255),
                    life_time=100)

        for i in range(n_vehicles):
            blueprint = np.random.choice(vehicle_blueprints)
            blueprint.set_attribute('role_name', 'autopilot')

            if blueprint.has_attribute('color'):
                blueprint.set_attribute(
                        'color',
                        np.random.choice(blueprint.get_attribute('color').recommended_values))

            if blueprint.has_attribute('driver_id'):
                blueprint.set_attribute(
                        'driver_id',
                        np.random.choice(blueprint.get_attribute('driver_id').recommended_values))

            for _ in range(n_retries):
                # vehicle = self._world.try_spawn_actor(blueprint, spawn_points[np.random.choice(SPAWNS)])
                vehicle = self._world.try_spawn_actor(blueprint, np.random.choice(spawn_points))

                if vehicle is not None:
                    vehicle.set_autopilot(True)

                    self._actor_dict['vehicle'].append(vehicle)
                    self._vehicle_speeds.append(np.random.randint(MAX_SPEED_MS // 2, MAX_SPEED_MS + 1))
                    break

    def _spawn_pedestrians(self, n_pedestrians):
        SpawnActor = carla.command.SpawnActor

        peds_spawned = 0

        walkers = []

        while peds_spawned < n_pedestrians:
            spawn_points = []
            _walkers = []

            for i in range(n_pedestrians - peds_spawned):
                spawn_point = carla.Transform()
                loc = self._world.get_random_location_from_navigation()

                if loc is not None:
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)

            blueprints = self._blueprints.filter('walker.pedestrian.*')
            batch = []
            for spawn_point in spawn_points:
                walker_bp = random.choice(blueprints)

                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')

                batch.append(SpawnActor(walker_bp, spawn_point))

            for result in self._client.apply_batch_sync(batch, True):
                if not result.error:
                    peds_spawned += 1
                    _walkers.append(result.actor_id)

            walkers.extend(_walkers)

        self._actor_dict['pedestrian'].extend(self._world.get_actors(walkers))

    def _set_weather(self, weather_string):
        if weather_string == 'random':
            weather = np.random.choice(WEATHERS)
        else:
            weather = weather_string

        self.weather = weather
        self._world.set_weather(weather)

    def reset(self, start=0, weather='random', n_vehicles=10, n_pedestrians=10, seed=0):
        is_ready = False

        while not is_ready:
            np.random.seed(seed)

            self._clean_up()
            # self._spawn_player(self._map.get_spawn_points()[np.random.choice(SPAWNS)])
            self._spawn_player(np.random.choice(self._map.get_spawn_points()))
            self._setup_sensors()

            self._set_weather(weather)
            self._spawn_vehicles(n_vehicles)
            self._spawn_pedestrians(n_pedestrians)

            print('%d / %d vehicles spawned.' % (len(self._actor_dict['vehicle']), n_vehicles))
            print('%d / %d pedestrians spawned.' % (len(self._actor_dict['pedestrian']), n_pedestrians))

            is_ready = self.ready()

    def _spawn_player(self, start_pose):
        vehicle_bp = np.random.choice(self._blueprints.filter(VEHICLE_NAME))
        vehicle_bp.set_attribute('role_name', 'hero')

        self._player = self._world.spawn_actor(vehicle_bp, start_pose)

        map_utils.init(self._player)

        self._actor_dict['player'].append(self._player)

    def ready(self, ticks=10):
        self.step()

        for _ in range(ticks):
            self.step()

        with self._rgb_queue.mutex:
            self._rgb_queue.queue.clear()

        with self._seg_queue.mutex:
            self._seg_queue.queue.clear()

        self._time_start = time.time()
        self._tick = 0

        return not self.collided

    def step(self, control=None):
        if control is not None:
            self._player.apply_control(control)

        for i, vehicle in enumerate(self._actor_dict['vehicle']):
            if self._tick % 200 == 0:
                self._vehicle_speeds[i] = np.random.randint(MAX_SPEED_MS // 2, MAX_SPEED_MS + 1)

            max_speed = self._vehicle_speeds[i]
            velocity = vehicle.get_velocity()
            speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])

            if speed > max_speed:
                vehicle.set_velocity(vehicle.get_velocity() * 0.9)

        self._world.tick()
        map_utils.tick()

        self._tick += 1

        # Put here for speed (get() busy polls queue).
        rgb = None

        while rgb is None or self._rgb_queue.qsize() > 0:
            rgb = self._rgb_queue.get()

        seg = None

        while seg is None or self._seg_queue.qsize() > 0:
            seg = self._seg_queue.get()

        result = map_utils.get_observations()
        result.update({
            'collided': self.collided,
            'rgb': _carla_img_to_numpy(rgb),
            'segmentation': _carla_img_to_numpy(seg)[:, :, 0],
            'wall': time.time() - self._time_start,
            't': self._tick,
            })

        return result

    def _clean_up(self):
        for vehicle in self._actor_dict['vehicle']:
            vehicle.destroy()

        for sensor in self._actor_dict['sensor']:
            sensor.destroy()

        for actor_type in list(self._actor_dict.keys()):
            self._client.apply_batch([carla.command.DestroyActor(x) for x in self._actor_dict[actor_type]])
            self._actor_dict[actor_type].clear()

        self._actor_dict.clear()
        self._vehicle_speeds = list()

        self._tick = 0
        self._time_start = time.time()

        self._player = None

        # Clean-up cameras
        if self._rgb_queue:
            with self._rgb_queue.mutex:
                self._rgb_queue.queue.clear()

        if self._seg_queue:
            with self._seg_queue.mutex:
                self._seg_queue.queue.clear()

    def _setup_sensors(self):
        """
        Add sensors to _actor_dict to be cleaned up.
        """
        # Camera.
        self._rgb_queue = queue.Queue()

        rgb_camera_bp = self._blueprints.find('sensor.camera.rgb')
        rgb_camera_bp.set_attribute('image_size_x', '384')
        rgb_camera_bp.set_attribute('image_size_y', '160')
        rgb_camera_bp.set_attribute('fov', '90')
        rgb_camera = self._world.spawn_actor(
            rgb_camera_bp,
            carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=0)),
            attach_to=self._player)
        rgb_camera.listen(self._rgb_queue.put)

        self._actor_dict['sensor'].append(rgb_camera)

        self._seg_queue = queue.Queue()

        seg_camera_bp = self._blueprints.find('sensor.camera.semantic_segmentation')
        seg_camera_bp.set_attribute('image_size_x', '384')
        seg_camera_bp.set_attribute('image_size_y', '160')
        seg_camera_bp.set_attribute('fov', '90')

        seg_camera = self._world.spawn_actor(
            seg_camera_bp,
            carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=0)),
            attach_to=self._player)
        seg_camera.listen(self._seg_queue.put)

        self._actor_dict['sensor'].append(seg_camera)

        # Collisions.
        self.collided = False
        self._collided_frame_number = -1

        collision_sensor = self._world.spawn_actor(
                self._blueprints.find('sensor.other.collision'),
                carla.Transform(), attach_to=self._player)
        collision_sensor.listen(lambda event: self.__class__._on_collision(weakref.ref(self), event))

        self._actor_dict['sensor'].append(collision_sensor)

    @staticmethod
    def _on_collision(weakself, event):
        _self = weakself()

        if not _self:
            return

        impulse = event.normal_impulse
        intensity = np.linalg.norm([impulse.x, impulse.y, impulse.z])

        print(intensity)

        if intensity > COLLISION_THRESHOLD:
            _self.collided = True
            _self._collided_frame_number = event.frame_number

    def __enter__(self):
        set_sync_mode(self._client, True)

        return self

    def __exit__(self, *args):
        """
        Make sure to set the world back to async,
        otherwise future clients might have trouble connecting.
        """
        self._clean_up()

        set_sync_mode(self._client, False)
