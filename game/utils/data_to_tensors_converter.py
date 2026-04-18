import numpy as np

from settings import GAN_TILE_DICT, ROOM_TILE_DICT, ROOM_TYPES


WALL = ROOM_TILE_DICT['WALL']
FLOOR = ROOM_TILE_DICT['FLOOR']
DOOR = ROOM_TILE_DICT['DOOR']
ENEMY = ROOM_TILE_DICT['ENEMY']
CHEST = ROOM_TILE_DICT['CHEST']
HEALING = ROOM_TILE_DICT['HEALING']

CHANNEL_MAP = {
    WALL: 0,
    FLOOR: 1,
    DOOR: 2,
    ENEMY: 3,
    CHEST: 4,
    HEALING: 5
}

# CHANNEL_MAP = {
#     EMPTY: 0,
#     ENEMY: 1,
#     CHEST: 2,
#     HEALING: 3
# }

NUM_CHANNELS = len(CHANNEL_MAP)

def room_to_tensor(room_matrix):
    '''
    Converts a room matrix (2D list of tile integers) into a 3D tensor with one-hot encoding across channels.
    '''
    height = len(room_matrix)
    width = len(room_matrix[0])

    # Initialize tensor with zeros
    tensor = np.zeros((NUM_CHANNELS, height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            tile = room_matrix[y][x]

            if tile in CHANNEL_MAP:
                channel = CHANNEL_MAP[tile]
                tensor[channel, y, x] = 1.0

    return tensor

def dataset_to_tensors(dataset):
    '''
    Converts a list of room matrices into a numpy array of tensors suitable for GAN training.
    '''
    tensors = []
    labels = []

    for i, entry in enumerate(dataset):
        room_matrix = next(v for k, v in entry.items() if k.startswith("room_matrix"))
        room_type = entry['type']

        tensor = room_to_tensor(room_matrix)
        tensors.append(tensor)

        labels.append(ROOM_TYPES[room_type])

    return (np.array(tensors, dtype=np.float32), np.array(labels, dtype=np.int64))

# def normalize(data):
#     '''
#     Normalizes data from [0,1] to [-1,1] for GAN training.
#     '''
#     return data * 2 - 1

