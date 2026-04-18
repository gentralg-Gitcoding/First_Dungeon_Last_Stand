import time
import numpy as np

from utils.save_load_data import load_json_dataset, save_tensor_dataset
from settings import ROOM_TILE_DICT, ROOM_TYPES


DATA_SYNTH_PATH = 'game/data/synthetic_rooms_dataset.json'
DATA_TENSOR_PATH = 'game/data/diffusion_tensors.npz'

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
    room_matrix = np.array(room_matrix)
    height, width = room_matrix.shape

    # Initialize tensor with zeros
    tensor = np.zeros((NUM_CHANNELS, height, width), dtype=np.float32)

    for tile, channel in CHANNEL_MAP.items():
        tensor[channel] = (room_matrix == tile).astype(np.float32)

    return tensor

def dataset_to_tensors(dataset):
    '''
    Converts a list of room matrices into a numpy array of tensors suitable for Diffusion training.
    '''
    tensors = []
    labels = []

    for i, entry in enumerate(dataset):
        room_matrix = next(v for k, v in entry.items() if k.startswith("room_matrix"))
        room_type = entry['type']

        tensor = room_to_tensor(room_matrix)
        tensors.append(tensor)

        labels.append(ROOM_TYPES[room_type])

        normalized_data = normalize(np.array(tensors, dtype=np.int8))

    return (normalized_data, np.array(labels, dtype=np.int64))

def normalize(tensors):
    '''
    Normalizes data from [0,1] to [-1,1] for Diffuser training.
    '''
    return tensors * 2 - 1

def main():
    start = time.time()
    dataset = load_json_dataset(DATA_SYNTH_PATH)

    start = time.time()
    tensors, labels = dataset_to_tensors(dataset)

    start = time.time()
    save_tensor_dataset(tensors, labels, DATA_TENSOR_PATH)

    print("Saved:", time.time() - start)

if __name__ == "__main__":
    main()