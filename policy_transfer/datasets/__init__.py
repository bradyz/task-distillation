from . import vizdoom_dataset
from . import supertux_dataset
from . import carla_dataset


SOURCES = {
        'vizdoom': vizdoom_dataset.get_dataset,
        'supertux': supertux_dataset.get_dataset,
        'carla': carla_dataset.get_dataset,
        }


def get_dataset(source):
    return SOURCES[source]
