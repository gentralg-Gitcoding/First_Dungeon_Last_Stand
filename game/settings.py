SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

#Makes each tile 32x32 pixels
TILE_SIZE = 32

#Size of the map layout array
MAP_WIDTH = 30
MAP_HEIGHT = 20

#limits frame rates
FPS = 60

#Room size
ROOM_WIDTH = SCREEN_WIDTH // TILE_SIZE
ROOM_HEIGHT = SCREEN_HEIGHT // TILE_SIZE

#Max rooms until boss
MAX_ROOMS = 10

# Legend
ROOM_TILE_DICT = {
    'WALL': 0,   # wall
    'FLOOR': 1,   # floor
    'DOOR': 2,   # door
    'ENEMY': 3,   # enemy
    'CHEST': 4,   # chest
    'HEALING': 5,    # healing
}

GAN_TILE_DICT = {
    'EMPTY': 0,   # empty
    'ENEMY': 1,   # enemy
    'CHEST': 2,   # chest
    'HEALING' : 3    # healing
}

ROOM_TYPES = {
    "enemy": 0,
    "loot": 1,
    "healing": 2,
    "start": 3,
    "boss": 4,
}

GAN_TO_ROOM_TILE = {
    0: 'empty',
    1: 'enemy',
    2: 'chest',
    3: 'healing',
}

MATRIX_TO_ROOM_TILE = {
    0: 'wall',   # wall
    1: 'floor',   # floor
    2: 'door',   # door
    3: 'enemy',   # enemy
    4: 'chest',   # chest
    5: 'healing'    # healing
}

DIRECTION_VECTORS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0)
}
