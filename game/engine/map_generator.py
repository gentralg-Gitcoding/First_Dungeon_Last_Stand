import random
import json
import copy

import numpy as np
from engine.gan_generator import Generator, generate_room
from ai.diffusion_generator import SimpleUNet, generate_diffusion_dungeon_room
from utils.data_to_dataloader_converter import get_dataloader
from settings import GAN_TILE_DICT, ROOM_HEIGHT, ROOM_WIDTH, MAX_ROOMS, ROOM_TILE_DICT, MATRIX_TO_ROOM_TILE, ROOM_TYPES
from utils.save_load_data import load_json_dataset
import torch 





# NOTE: This bool flag is for running the game with synthetic data for testing purposes, DO NOT KEEP THIS IN FINAL GAME, ONLY FOR TESTING
# options are "testing" or "controlled" or ""
output_type = ''

#NOTE: DO NOT KEEP, TESTING ONLY
if output_type == "testing":
    DATASET = load_json_dataset('game/data/synthetic_rooms_dataset.json')


WALL = 0  
FLOOR = 1   
DOOR = 2  
ENEMY = 3   
CHEST = 4  
HEALING = 5

GAN_PATH = "game/data/models/generator_epoch_49.pth"
DIFFUSION_PATH = "game/data/models/diffusion_model.pth"

# Room Tracker
rooms = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_selection = "diffusion"  # Change to "gan" to use the GAN model instead

if model_selection == "gan":
    GENERATOR = Generator(noise_dim=100, num_room_types=len(ROOM_TYPES))
    state_dict = torch.load(GAN_PATH, map_location=torch.device('cpu'))
elif model_selection == "diffusion":
    GENERATOR = SimpleUNet(in_channels=6, num_room_types=len(ROOM_TYPES))
    state_dict = torch.load(DIFFUSION_PATH, map_location=torch.device('cpu'))

if state_dict:
    GENERATOR.load_state_dict(state_dict)
    GENERATOR.to(device)
    print(f"Loaded {model_selection} model from {GAN_PATH if model_selection == 'gan' else DIFFUSION_PATH}")


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
    print(f"Extracting room matrix with values from ROOM_TILE_DICT: {ROOM_TILE_DICT}")

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
    # print(f"Applying matrix to room tiles using MATRIX_TO_ROOM_TILE: {MATRIX_TO_ROOM_TILE}")

    for y in range(room.h):
        for x in range(room.w):
            value = matrix[y][x]
            room.room_map[y][x] = MATRIX_TO_ROOM_TILE.get(value, 'WALL')

    return room


def create_structure_mask(room_matrix):
    '''
    Create a zero filled matrix and place a 1 on the matrix's walls, doors and edges to force rules for models to not modify.
    '''
    height = len(room_matrix)
    width = len(room_matrix[0])

    mask = [[0 for _ in range(width)] for _ in range(height)]

    for y in range(height):
        for x in range(width):

            tile = room_matrix[y][x]

            # Protect doors let the models modify the interior of the room but not the doors to ensure connectivity is always maintained
            if tile == DOOR:  
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
                if entity == WALL:     #Wall
                    original[y][x] = WALL
                elif entity == ENEMY:     #Enemy
                    original[y][x] = ENEMY  
                elif entity == CHEST:     #Chest
                    original[y][x] = CHEST
                elif entity == HEALING:     #Healing
                    original[y][x] = HEALING
                # else:
                #     original[y][x] = FLOOR

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
    '''
    Apply constraints to the generated room matrix to enforce stronger bias towards the assigned room type, can be used as a post process after generation. 
    For example, an "enemy" room should have more enemies and less chests/healing, while a "healing" room should have more healing fountains and less enemies.
    '''
    removal_count = 0

    for y in range(len(matrix)):
        for x in range(len(matrix[0])):

            if room_type == "enemy":
                if matrix[y][x] in [CHEST, HEALING]:  # chest or healing fountain
                    matrix[y][x] = WALL
                    removal_count += 1

            elif room_type == "loot":
                if matrix[y][x] in [ENEMY, HEALING]:  # enemy or healing fountain
                    matrix[y][x] = WALL
                    removal_count += 1

            elif room_type == "healing":
                if matrix[y][x] in [ENEMY, CHEST]:   # enemy or chest 
                    matrix[y][x] = WALL
                    removal_count += 1

    print(f"Enforced {removal_count} tile removals for room type bias towards {room_type} room.")
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

def remove_trapped_enemies(matrix):
    '''Remove enemies that are completely encased by walls to prevent unfair spawns, can be used as a post process after all constraints are applied.'''
    H = len(matrix)
    W = len(matrix[0])
    count = 0

    for y in range(1, H-1):
        for x in range(1, W-1):
            if matrix[y][x] == ENEMY:
                neighbors = [
                    matrix[y+1][x], matrix[y-1][x],
                    matrix[y][x+1], matrix[y][x-1]
                ]
                if all(n == WALL for n in neighbors):
                    count += 1
                    matrix[y][x] = FLOOR

    print(f"Saved {count} trapped enemies.")
    return matrix

def clean_generated_doors(matrix, original_matrix):
    height = len(matrix)
    width = len(matrix[0])
    count = 0

    for y in range(height):
        for x in range(width):

            # If it's a door in generated output
            if matrix[y][x] == DOOR:

                # Keep ONLY if it was originally a door
                if original_matrix[y][x] != DOOR:
                    matrix[y][x] = FLOOR
                    count += 1
    print(f"{count} Doors cleaned from generated output.")
    return matrix

def enforce_entity_limits(matrix, room_type):

    limits = {
        "loot": {4: (1, 3)},
        "healing": {5: (1, 2)},
        "enemy": {3: (24, 48)}
    }
    if limits.get(room_type, {}) == {}:
        return Room(0, 0, ROOM_WIDTH, ROOM_HEIGHT).room_map

    type_limits = limits.get(room_type, {})
    added_count = 0
    removed_count = 0

    for tile_type, (min_limit, max_limit) in type_limits.items():
        count = np.sum(matrix == tile_type)     # Count current entities of this type

        # If we are over max limit, start removing entities randomly until we are under the max limit
        if count > max_limit:
            tile_positions = np.argwhere(matrix == tile_type) # Get all positions of this tile type

            np.random.shuffle(tile_positions)  # Shuffle to add randomness to removal

            for y, x in tile_positions[:(count - max_limit)]:
                matrix[y][x] = FLOOR
                count -= 1
                removed_count += 1


        # Hard enforce min limits if we are under the minimum limit by placing entities randomly until we reach the minimum limit
        elif count < min_limit:
            # Get all valid floor positions for potential entity placement, excluding edges to prevent unfair placements
            floor_positions = [
                (y, x) for y, x in np.argwhere(matrix == FLOOR)
                if 0 < y < matrix.shape[0]-1 and 0 < x < matrix.shape[1]-1
            ]

            np.random.shuffle(floor_positions)  # Shuffle to add randomness to placement

            for y, x in floor_positions[:(min_limit - count)]:
                matrix[y][x] = tile_type
                added_count += 1


    print(f"Initially had {count} entities in {room_type} room.")
    print(f"Added {added_count} entities and removed {removed_count} entities.")
    return matrix


def get_noise_schedule(T=200, device="cpu"):
    beta_start = 1e-4
    beta_end = 0.02

    betas = torch.linspace(beta_start, beta_end, T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return alphas_cumprod

def tensor_to_tilemap(tensor):
    '''Convert diffusion output tensor into tilemap matrix of tile indices'''
    print("Incoming tensor shape:", tensor.shape)
    tensor = tensor.squeeze(0)              # remove batch

    matrix = torch.argmax(tensor, dim=0)    # (H, W)
    print("After argmax:", matrix.shape)
    matrix = matrix.cpu().numpy()
    return matrix



def generate_dungeon_room(width = ROOM_WIDTH, height = ROOM_HEIGHT):
    room = Room(0, 0, width, height)

    #NOTE: Test premade maps, DO NOT KEEP THIS
    if output_type == "testing":
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

        if model_selection == "gan":
            # Generate GAN created room
            gan_matrix = generate_room(GENERATOR, room.type, room.w, room.h)

            if output_type == "controlled":

                # Apply constraints to control the room type 
                gan_enforced_matrix = enforce_room_type_bias(gan_matrix, room.type)

                # Apply cross entity transform turning GAN matrix values into room matrix values
                entity_matrix = apply_entities(room.room_map, gan_enforced_matrix, mask)

                # Remove walls encasing enemies to prevent unfair spawns
                entity_matrix = remove_trapped_enemies(entity_matrix)

                entity_matrix = enforce_entity_limits(entity_matrix, room.type)

                # Connectivity Check to make sure all doors have room for movement
                final_matrix = enforce_reachable_door(entity_matrix)

                # Draw final matrix transform into room tile characters
                final_room = apply_matrix_to_room_tiles(room, final_matrix)

                return final_room
            else:
                final_room = apply_matrix_to_room_tiles(room, gan_matrix)
                return final_room

        elif model_selection == "diffusion":
            # Generate Diffusion created room
            alphas_cumprod = get_noise_schedule(device=device)

            diff_tensor = generate_diffusion_dungeon_room(
                GENERATOR,
                ROOM_TYPES[room.type],
                room.room_map,
                mask,
                alphas_cumprod,
                device
            )

            # tensor_to_tilemap
            diff_matrix = diff_tensor.squeeze(0).cpu().numpy() # (H, W)

            unique, counts = np.unique(diff_matrix, return_counts=True)
            print(f"Tensor to Tilemaps Matrix Counts: {dict(zip(unique, counts))}")

            if output_type == "controlled":

                # enforce structure (walls + doors)
                for y in range(len(room.room_map)):
                    for x in range(len(room.room_map[0])):
                        if mask[y][x] == 1:     # Only modify non-locked tiles
                            diff_matrix[y][x] = room.room_map[y][x]


                # Post process diffusion output with same constraints as GAN to control room type and ensure playability
                diff_matrix = enforce_room_type_bias(diff_matrix, room.type)
                diff_matrix = clean_generated_doors(diff_matrix, room.room_map)
                diff_matrix = remove_trapped_enemies(diff_matrix)
                diff_matrix = enforce_entity_limits(diff_matrix, room.type)
                diff_matrix = enforce_reachable_door(diff_matrix)

                # Draw final matrix transform into room tile characters
                final_room = apply_matrix_to_room_tiles(room, diff_matrix)

                return final_room
            
            else:
                final_room = apply_matrix_to_room_tiles(room, diff_matrix)
                return final_room

    else:
        #Return original room (start or boss)
        final_room = apply_matrix_to_room_tiles(room, room.room_map)

        return final_room
