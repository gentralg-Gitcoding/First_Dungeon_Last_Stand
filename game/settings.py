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

WALL = 'WALL'
FLOOR = 'FLOOR'
DOOR = 'DOOR'
ENEMY = 'ENEMY'
CHEST = 'CHEST'
HEALING = 'HEALING'
EMPTY = 'EMPTY' 

# Legend
ROOM_TILE_DICT = {
    WALL: 0,   # wall
    FLOOR: 1,   # floor
    DOOR: 2,   # door
    ENEMY: 3,   # enemy
    CHEST: 4,   # chest
    HEALING: 5    # healing
}

GAN_TILE_DICT = {
    EMPTY: 0,   # empty
    ENEMY: 1,   # enemy
    CHEST: 2,   # chest
    HEALING : 3    # healing
}

ROOM_TYPES = {
    "enemy": 0,
    "loot": 1,
    "healing": 2,
    "start": 3,
    "boss": 4,
}

GAN_TO_ROOM_TILE = {
    0: EMPTY,
    1: ENEMY,
    2: CHEST,
    3: HEALING,
}

MATRIX_TO_ROOM_TILE = {
    0: WALL,   # wall
    1: FLOOR,   # floor
    2: DOOR,   # door
    3: ENEMY,   # enemy
    4: CHEST,   # chest
    5: HEALING    # healing
}

