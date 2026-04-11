import random
import json
from collections import deque
from engine.gan_generator import Generator, generate_room
from settings import ROOM_HEIGHT, ROOM_WIDTH, MAX_ROOMS, ROOM_TILE_DICT, MATRIX_TO_ROOM_TILE

# Room Tracker
rooms = []

GENERATOR = Generator()
TESTING_SYNTH = True
DATA_PATH = 'game/data/synthetic_rooms_dataset.json'

def load_dataset(path=DATA_PATH):
    with open(path, "r") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} rooms")
    return dataset

#NOTE: DO NOT KEEP, TESTING ONLY
if TESTING_SYNTH:
    DATASET = load_dataset()


class Room:
    #Starting point: x, y
    #Area lengths: width and heights w, h
    def __init__(self, x, y, w, h):
        self.x = x     
        self.y = y
        self.w = w
        self.h = h
        self.type = None

        self.room_map = [["#" for _ in range(w)] for _ in range(h)]

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                self.room_map[y][x] = "."

    def center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)

    def intersects(self, other):
        return (
            self.x < other.x + other.w and
            self.x + self.w > other.x and
            self.y < other.y + other.h and
            self.y + self.h > other.y
        )


def place_doors(room_map):
    height = len(room_map)
    width = len(room_map[0])
    doors = []

    # Top
    room_map[0][width // 2] = '+'
    doors.append(("top", width // 2, 0))

    # Bottom
    room_map[height - 1][width // 2] = '+'
    doors.append(("bottom", width // 2, height - 1))

    # Left
    room_map[height // 2][0] = '+'
    doors.append(("left", 0, height // 2))

    # Right
    room_map[height // 2][width - 1] = '+'
    doors.append(("right", width - 1, height // 2))

    return doors


def assign_room_type(room):
    global rooms
    rooms.append(room)

    for i, room in enumerate(rooms):
        #Check if room already was assigned, skip it
        if room.type:
            continue

        if i == 0:
            room.type = "start"
        elif i % MAX_ROOMS == 0:
            room.type = "boss"
        else:
            r = random.random()     #Weight for room type

            if r < 0.1:
                room.type = "healing"
            elif r < 0.2:
                room.type = "loot"
            else:
                room.type = "enemy"


def extract_room_matrix(room):
    '''
    Get Numerical representation of room tiles as a matrix, used mainly for GANs learning
    '''
    matrix = []

    for y in range(room.y, room.h):
        row = []
        for x in range(room.x, room.w):
            tile = room.room_map[y][x]
            row.append(ROOM_TILE_DICT.get(tile, 0))
        matrix.append(row)

    return matrix


def apply_matrix_to_room_tiles(room, matrix):
    '''
    Convert numerical matrices into room tile characters from the MATRIX_TO_ROOM_TILE dict in settings
    '''
    for y in range(room.h):
        for x in range(room.w):
            value = matrix[y][x]
            room.room_map[y][x] = MATRIX_TO_ROOM_TILE.get(value, '#')

    return room


def create_structure_mask(room_matrix):
    '''
    Create a zero filled matrix and place a 1 on the matrix's walls, doors and edges to force rules for GAN model to not modify.
    '''
    height = len(room_matrix)
    width = len(room_matrix[0])

    mask = [[0 for _ in range(width)] for _ in range(height)]

    for y in range(height):
        for x in range(width):

            tile = room_matrix[y][x]

            # Protect walls and doors
            if tile == 0 or tile == 2:  # wall or door
                mask[y][x] = 1  # LOCKED

            # Protect room edges
            elif (
                x == 0 or x == width - 1 or
                y == 0 or y == height - 1
            ):
                mask[y][x] = 1

    return mask


def apply_entities(original, generated, mask, density=0.2):
    '''
    Cycle through tiles and find safe tiles for GANs to safely modify any changes made. 
    '''
    height = len(original)
    width = len(original[0])

    for y in range(height):
        for x in range(width):

            # Only place on FLOOR tiles
            if original[y][x] != 1:
                continue

            #Control how heavily we want to modify the room
            if random.random() > density:
                continue

            # 0 masked as safe, give it to GANs
            if mask[y][x] == 0:     
                entity = generated[y][x]

                #Convert GANs dict values into room dict values
                if entity == 0:     #Wall
                    original[y][x] = 0
                if entity == 1:     #Enemy
                    original[y][x] = 3
                if entity == 2:     #Chest
                    original[y][x] = 4
                if entity == 3:     #Healing
                    original[y][x] = 5

    return original


def enforce_reachable_door(matrix):
    height = len(matrix)
    width = len(matrix[0])

    for y in range(height):
        for x in range(width):

            # Check if tile is a door
            if matrix[y][x] == 2:

                # TOP EDGE
                if y == 0:
                    if y + 1 < height:
                        matrix[y + 1][x] = 1  # floor

                # BOTTOM EDGE
                elif y == height - 1:
                    if y - 1 >= 0:
                        matrix[y - 1][x] = 1

                # LEFT EDGE
                elif x == 0:
                    if x + 1 < width:
                        matrix[y][x + 1] = 1

                # RIGHT EDGE
                elif x == width - 1:
                    if x - 1 >= 0:
                        matrix[y][x - 1] = 1

    return matrix

def enforce_room_type_bias(matrix, room_type):

    for y in range(len(matrix)):
        for x in range(len(matrix[0])):

            if room_type == "enemy":
                if matrix[y][x] in [2,3]:  # chest or healing fountain
                    matrix[y][x] = 0

            elif room_type == "loot":
                if matrix[y][x] in [1,3]:  # enemy or healing fountain
                    matrix[y][x] = 0

            elif room_type == "healing":
                if matrix[y][x] in [1,2]:   # enemy or chest 
                    matrix[y][x] = 0

    return matrix


def generate_dungeon_room(width = ROOM_WIDTH, height = ROOM_HEIGHT):
    room = Room(0, 0, width, height)

    #NOTE: Test premade maps, DO NOT KEEP THIS
    if TESTING_SYNTH:
        room.room_map, room.type = dict.values(random.choice(DATASET))
        room = apply_matrix_to_room_tiles(room, room.room_map)
        return room

    #Assign the type of room
    assign_room_type(room)

    #Assign doors in the room map
    place_doors(room.room_map)

    #Skip GAN for start and boss rooms
    if room.type not in ["start", "boss"]:
        # Get room matrix of the map
        room_matrix = extract_room_matrix(room)

        # Create structure mask
        mask = create_structure_mask(room_matrix)

        # Generate GAN created room
        gan_matrix = generate_room(GENERATOR, room.type, room.w, room.h)

        # Apply constraints to control the GAN
        gan_enforced_matrix = enforce_room_type_bias(gan_matrix, room.type)

        # Apply cross entity transform turning GAN matrix values into room matrix values
        entity_matrix = apply_entities(room_matrix, gan_enforced_matrix, mask)

        # Connectivity Check to make sure all doors have room for movement
        final_matrix = enforce_reachable_door(entity_matrix)

        # Draw final matrix transform into room tile characters
        room = apply_matrix_to_room_tiles(room, final_matrix)

    return room