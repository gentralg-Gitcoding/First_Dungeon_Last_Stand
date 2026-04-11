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
    "#": 0,   # wall
    ".": 1,   # floor
    "+": 2,   # door
    "E": 3,   # enemy
    "C": 4,   # chest
    "H": 5    # healing
}

GAN_TILE_DICT = {
    "#": 0,   # wall
    "E": 1,   # enemy
    "C": 2,   # chest
    "H": 3    # healing
}

ROOM_TYPES = {
    "enemy": 0,
    "loot": 1,
    "healing": 2,
    "start": 3,
    "boss": 4,
}

GAN_TO_ROOM_TILE = {
    0: "#",
    1: "E",
    2: "C",
    3: "H"
}

MATRIX_TO_ROOM_TILE = {
    0:"#",   # wall
    1:".",   # floor
    2:"+",   # door
    3:"E",   # enemy
    4:"C",   # chest
    5:"H"    # healing
}