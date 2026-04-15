import random
import json
import copy
from engine.gan_generator import Generator, generate_room
from settings import GAN_TILE_DICT, ROOM_HEIGHT, ROOM_WIDTH, MAX_ROOMS, ROOM_TILE_DICT, MATRIX_TO_ROOM_TILE, ROOM_TYPES
from utils.save_load_data import load_json_dataset
import torch 

WALL = 0  
FLOOR = 1   
DOOR = 2  
ENEMY = 3   
CHEST = 4  
HEALING = 5    
EMPTY = 0

# Room Tracker
rooms = []

GENERATOR = Generator(noise_dim=100, num_room_types=len(ROOM_TYPES))
state_dict = torch.load("game/data/models/generator_epoch_49.pth", map_location=torch.device('cpu'))
if state_dict:
    GENERATOR.load_state_dict(state_dict)

TESTING_SYNTH = False
DATA_SYNTH_PATH = 'game/data/synthetic_rooms_dataset.json'

#NOTE: DO NOT KEEP, TESTING ONLY
if TESTING_SYNTH:
    DATASET = load_json_dataset(DATA_SYNTH_PATH)


class Room:
    #Starting point: x, y
    #Area lengths: width and heights w, h
    def __init__(self, x, y, w, h):
        self.x = x     
        self.y = y
        self.w = w
        self.h = h
        self.type = None

        self.room_map = [[0 for _ in range(w)] for _ in range(h)]

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                self.room_map[y][x] = 1

        self.doors = self.place_doors(self.room_map)

    def center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)

    def intersects(self, other):
        return (
            self.x < other.x + other.w and
            self.x + self.w > other.x and
            self.y < other.y + other.h and
            self.y + self.h > other.y
        )

    def place_doors(self, room_map):
        height = len(room_map)
        width = len(room_map[0])
        doors = []

        # Top
        room_map[0][width // 2] = 2
        doors.append(("top", width // 2, 0))

        # Bottom
        room_map[height - 1][width // 2] = 2
        doors.append(("bottom", width // 2, height - 1))

        # Left
        room_map[height // 2][0] = 2   
        doors.append(("left", 0, height // 2))

        # Right
        room_map[height // 2][width - 1] = 2
        doors.append(("right", width - 1, height // 2))

        return doors


def assign_room_type(room):
    global rooms

    rooms.append(room)
    i = len(rooms) - 1  #Check last room 

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
            room.room_map[y][x] = MATRIX_TO_ROOM_TILE.get(value, 'WALL')

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
            if tile == WALL or tile == DOOR:  # wall or door
                mask[y][x] = 1  # LOCKED

            # Protect room edges
            elif (
                x == 0 or x == width - 1 or
                y == 0 or y == height - 1
            ):
                mask[y][x] = 1

    return mask


def apply_entities(original, generated, mask, density=0.6):
    '''
    Cycle through tiles and find safe tiles for GANs to safely modify any changes made. 
    '''
    height = len(original)
    width = len(original[0])

    for y in range(height):
        for x in range(width):

            # Only place on FLOOR tiles
            if original[y][x] != FLOOR:
                continue

            #Control how heavily we want to modify the room
            if random.random() > density:
                continue

            # 0 masked as safe, give it to GANs
            if mask[y][x] == 0:     
                entity = generated[y][x]

                #Convert GANs dict values into room dict values
                # if entity == WALL:     #Wall
                #     original[y][x] = WALL
                if entity == ENEMY:     #Enemy
                    original[y][x] = ENEMY  
                elif entity == CHEST:     #Chest
                    original[y][x] = CHEST
                elif entity == HEALING:     #Healing
                    original[y][x] = HEALING
                else:
                    original[y][x] = FLOOR

    return original


def enforce_reachable_door(matrix):
    height = len(matrix)
    width = len(matrix[0])

    for y in range(height):
        for x in range(width):

            # Check if tile is a door
            if matrix[y][x] == DOOR:

                # TOP EDGE
                if y == 0:
                    if y + 1 < height:
                        matrix[y + 1][x] = FLOOR

                # BOTTOM EDGE
                elif y == height - 1:
                    if y - 1 >= 0:
                        matrix[y - 1][x] = FLOOR

                # LEFT EDGE
                elif x == 0:
                    if x + 1 < width:
                        matrix[y][x + 1] = FLOOR

                # RIGHT EDGE
                elif x == width - 1:
                    if x - 1 >= 0:
                        matrix[y][x - 1] = FLOOR

    return matrix

def enforce_room_type_bias(matrix, room_type):

    for y in range(len(matrix)):
        for x in range(len(matrix[0])):

            if room_type == "enemy":
                if matrix[y][x] in [CHEST, HEALING]:  # chest or healing fountain
                    matrix[y][x] = WALL

            elif room_type == "loot":
                if matrix[y][x] in [ENEMY, HEALING]:  # enemy or healing fountain
                    matrix[y][x] = WALL

            elif room_type == "healing":
                if matrix[y][x] in [ENEMY, CHEST]:   # enemy or chest 
                    matrix[y][x] = WALL

    return matrix

def ensure_minimum_entities(matrix, room_type):
    count = sum(cell in [ENEMY, CHEST, HEALING] for row in matrix for cell in row)

    if count == 0:
        # Force at least one spawn
        h = len(matrix)
        w = len(matrix[0])

        for _ in range(10):
            y = random.randint(1, h-2)
            x = random.randint(1, w-2)

            if matrix[y][x] == FLOOR:
                if room_type == "enemy":
                    matrix[y][x] = ENEMY
                elif room_type == "loot":
                    matrix[y][x] = CHEST
                elif room_type == "healing":
                    matrix[y][x] = HEALING
                break

    return matrix


def boost_entities(matrix):
    '''
    Optional function to boost the number of entities in a room if the GAN is being too conservative, can be used as a post process after all constraints are applied.
    '''
    for y in range(len(matrix)):
        for x in range(len(matrix[0])):
            if matrix[y][x] == WALL:
                # convert some walls into entities
                r = random.random()
                if r < 0.05:
                    matrix[y][x] = ENEMY
                elif r < 0.08:
                    matrix[y][x] = CHEST
                elif r < 0.1:
                    matrix[y][x] = HEALING
    return matrix

def generate_dungeon_room(width = ROOM_WIDTH, height = ROOM_HEIGHT):
    room = Room(0, 0, width, height)

    #NOTE: Test premade maps, DO NOT KEEP THIS
    if TESTING_SYNTH:
        #Copy the dataset, DO NOT MODIFY THE DATA
        dataset_copy = random.choice(copy.deepcopy(DATASET))
        room.room_map, room.type = dict.values(dataset_copy)
        room = apply_matrix_to_room_tiles(room, room.room_map)
        print(f'\nSynth Room: {list(dataset_copy)[0]}')
        print(f'Room type: {room.type}')
        return room

    #Assign the type of room
    assign_room_type(room)

    #Skip GAN for start and boss rooms
    if room.type not in ["start", "boss"]:
        # Get room matrix of the map
        # room_matrix = extract_room_matrix(room)

        # Create structure mask
        mask = create_structure_mask(room.room_map)

        # Generate GAN created room
        gan_matrix = generate_room(GENERATOR, room.type, room.w, room.h)

        # Apply constraints to control the GAN
        gan_enforced_matrix = enforce_room_type_bias(gan_matrix, room.type)

        # Apply cross entity transform turning GAN matrix values into room matrix values
        entity_matrix = apply_entities(room.room_map, gan_enforced_matrix, mask)

        entity_matrix = ensure_minimum_entities(entity_matrix, room.type)

        # Connectivity Check to make sure all doors have room for movement
        final_matrix = enforce_reachable_door(entity_matrix)

        # Draw final matrix transform into room tile characters
        final_matrix = apply_matrix_to_room_tiles(room, final_matrix)

        return final_matrix
    else: 
        final_matrix = apply_matrix_to_room_tiles(room, room.room_map)

        return final_matrix