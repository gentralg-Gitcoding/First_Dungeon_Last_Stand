import os
import random
import json
import copy

from huggingface_hub import hf_hub_download
import numpy as np
from ai.gan_generator import Generator, generate_room
from ai.diffusion_generator import SimpleUNet, generate_diffusion_dungeon_room
from engine.entity_system import Chest, Enemy, Entity, HealingFountain
from utils.load_and_scale import load_and_scale
from utils.data_to_dataloader_converter import get_dataloader
from settings import ROOM_HEIGHT, ROOM_WIDTH, MAX_ROOMS, ROOM_TILE_DICT, ROOM_TYPES
from utils.save_load_data import load_json_dataset
import torch 





# NOTE: This bool flag is for running the game with synthetic data for testing purposes, DO NOT KEEP THIS IN FINAL GAME, ONLY FOR TESTING
OUTPUT_OPTIONS = [
    "controlled",
    "testing",
    ""
]
output_type = OUTPUT_OPTIONS[0]

# NOTE: DO NOT KEEP, USED FOR TESTING SYNTHETIC DATA
if output_type == "testing":
    DATASET = load_json_dataset('game/data/synthetic_rooms_dataset.json')




GAN_PATH = "game/data/models/generator_epoch_49.pth"
DIFFUSION_PATH = "game/data/models/diffusion_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Room Tracker
rooms = []
model_selection = "diffusion"  # Change to "gan" to use the GAN model instead

if model_selection == "gan":
    GENERATOR = Generator()
    if os.path.exists(GAN_PATH):
        state_dict = torch.load(GAN_PATH, map_location=DEVICE)
        print(f"Loaded {model_selection} model from {GAN_PATH}")
    else:
        hf_path = hf_hub_download(
            repo_id="gentralg/GANs-Dungeon-Floor-Entities",
            filename="generator_model.pth"
        )
        print(f"Loaded model from HF repo: {hf_path}")
        state_dict = torch.load(hf_path, map_location=DEVICE)
elif model_selection == "diffusion":
    GENERATOR = SimpleUNet()
    if os.path.exists(DIFFUSION_PATH):
        state_dict = torch.load(DIFFUSION_PATH, map_location=DEVICE)
        print(f"Loaded {model_selection} model from {DIFFUSION_PATH}")
    else:
        hf_path = hf_hub_download(
            repo_id="gentralg/Diffusion-Dungeon-Floor-Entities",
            filename="diffusion_model.pth"
        )
        print(f"Loaded model from HF repo: {hf_path}")
        state_dict = torch.load(hf_path, map_location=DEVICE)

if state_dict:
    GENERATOR.load_state_dict(state_dict)
    GENERATOR.to(DEVICE)


class Room:
    #Starting point: x, y
    #Area lengths: width and heights w, h
    def __init__(self, x, y, w, h):
        self.x = x     
        self.y = y
        self.w = w
        self.h = h
        self.type = None

        self.room_map = np.array([[0 for _ in range(w)] for _ in range(h)])

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                self.room_map[y][x] = 1

        self.doors = self.place_doors(self.room_map)

        self.entities = list[Entity]()

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

    def get_entity_at(self, x, y, exclude=None):
        for entity in self.entities:
            if entity == exclude:
                continue

            if entity.x == x and entity.y == y:
                return entity

        return None

    def tile_blocks_movement(self, x, y):
        tile = self.room_map[y][x]
        return tile == ROOM_TILE_DICT["WALL"]

    def entity_blocks_movement(self, x, y):
        entity = self.get_entity_at(x, y)
        if entity:
            return entity.blocks_movement

        return False
    
    def is_within_bounds(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h

    def is_blocked(self, x, y):
        '''Check if tile or entity blocks movement'''
        return (
            self.is_within_bounds(x, y) == False
            or self.tile_blocks_movement(x, y)
            or self.entity_blocks_movement(x, y)
        )

    def update_entity_position(self, entity, new_x, new_y):
        old_x, old_y = entity.x, entity.y
        self.room_map[old_y][old_x] = ROOM_TILE_DICT['FLOOR']  # Update old position to floor
        self.room_map[new_y][new_x] = ROOM_TILE_DICT[entity.name]  # Update new position to enemy (or appropriate tile type based on entity)

    def remove_entity(self, entity):
        if entity in self.entities:
            self.entities.remove(entity)
            self.room_map[entity.y][entity.x] = ROOM_TILE_DICT['FLOOR']  # Update tilemap to reflect entity removal


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


# def extract_room_matrix(room):
#     '''
#     Get Numerical representation of room tiles as a matrix, used mainly for GANs learning
#     '''
#     matrix = []
#     print(f"Extracting room matrix with values from ROOM_TILE_DICT: {ROOM_TILE_DICT}")

#     for y in range(room.y, room.h):
#         row = []
#         for x in range(room.x, room.w):
#             tile = room.room_map[y][x]
#             row.append(ROOM_TILE_DICT.get(tile, 0))
#         matrix.append(row)

#     return matrix


# def apply_matrix_to_room_tiles(room, matrix):
#     '''
#     Convert numerical matrices into room tile characters from the MATRIX_TO_ROOM_TILE dict in settings
#     '''
#     # print(f"Applying matrix to room tiles using MATRIX_TO_ROOM_TILE: {MATRIX_TO_ROOM_TILE}")

#     for y in range(room.h):
#         for x in range(room.w):
#             value = matrix[y][x]
#             room.room_map[y][x] = MATRIX_TO_ROOM_TILE.get(value, 'WALL')

#     return room


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
            if tile == ROOM_TILE_DICT['DOOR']:  
                mask[y][x] = 1  # LOCKED

            # Protect room edges
            elif (
                x == 0 or x == width - 1 or
                y == 0 or y == height - 1
            ):
                mask[y][x] = 1

    return mask


def apply_entities(room, generated, mask, density=0.6):
    '''
    Cycle through tiles and find safe tiles for GANs to safely modify any changes made. 
    '''
    height = len(room.room_map)
    width = len(room.room_map[0])
    room_map = room.room_map
    entities = room.entities

    for y in range(height):
        for x in range(width):

            # Only place on FLOOR tiles
            if room_map[y][x] != ROOM_TILE_DICT['FLOOR']:
                continue

            #Control how heavily we want to modify the room
            if random.random() > density:
                continue

            # 0 masked as safe, give it to GANs
            if mask[y][x] == 0:     
                entity = generated[y][x]

                #Convert GANs dict values into room dict values
                if entity == ROOM_TILE_DICT['WALL']:     #Wall
                    room_map[y][x] = ROOM_TILE_DICT['WALL']
                elif entity == ROOM_TILE_DICT['ENEMY']:     #Enemy
                    entities.append(Enemy(x, y, None))
                    room_map[y][x] = ROOM_TILE_DICT['ENEMY']
                elif entity == ROOM_TILE_DICT['CHEST']:     #Chest
                    entities.append(Chest(x, y, None))
                    room_map[y][x] = ROOM_TILE_DICT['CHEST']    
                elif entity == ROOM_TILE_DICT['HEALING']:     #Healing
                    entities.append(HealingFountain(x, y, None))
                    room_map[y][x] = ROOM_TILE_DICT['HEALING']

    return room_map, entities


def enforce_reachable_door(room):
    matrix = room.room_map
    height = len(matrix)
    width = len(matrix[0])

    for y in range(height):
        for x in range(width):

            # Check if tile is a door
            if matrix[y][x] == ROOM_TILE_DICT['DOOR']:

                # TOP EDGE
                if y == 0:
                    if y + 1 < height:
                        matrix[y + 1][x] = ROOM_TILE_DICT['FLOOR']

                # BOTTOM EDGE
                elif y == height - 1:
                    if y - 1 >= 0:
                        matrix[y - 1][x] = ROOM_TILE_DICT['FLOOR']

                # LEFT EDGE
                elif x == 0:
                    if x + 1 < width:
                        matrix[y][x + 1] = ROOM_TILE_DICT['FLOOR']

                # RIGHT EDGE
                elif x == width - 1:
                    if x - 1 >= 0:
                        matrix[y][x - 1] = ROOM_TILE_DICT['FLOOR']

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
                if matrix[y][x] in [ROOM_TILE_DICT['CHEST'], ROOM_TILE_DICT['HEALING']]:  # chest or healing fountain
                    matrix[y][x] = ROOM_TILE_DICT['WALL']
                    removal_count += 1

            elif room_type == "loot":
                if matrix[y][x] in [ROOM_TILE_DICT['ENEMY'], ROOM_TILE_DICT['HEALING']]:  # enemy or healing fountain
                    matrix[y][x] = ROOM_TILE_DICT['WALL']
                    removal_count += 1

            elif room_type == "healing":
                if matrix[y][x] in [ROOM_TILE_DICT['ENEMY'], ROOM_TILE_DICT['CHEST']]:   # enemy or chest 
                    matrix[y][x] = ROOM_TILE_DICT['WALL']
                    removal_count += 1

    print(f"Enforced {removal_count} tile removals for room type bias towards {room_type} room.")
    return matrix

def boost_entities(matrix):
    '''
    Optional function to boost the number of entities in a room if the GAN is being too conservative, can be used as a post process after all constraints are applied.
    '''
    for y in range(len(matrix)):
        for x in range(len(matrix[0])):
            if matrix[y][x] == ROOM_TILE_DICT['WALL']:
                # convert some walls into entities
                r = random.random()
                if r < 0.05:
                    matrix[y][x] = ROOM_TILE_DICT['ENEMY']
                elif r < 0.08:
                    matrix[y][x] = ROOM_TILE_DICT['CHEST']
                elif r < 0.1:
                    matrix[y][x] = ROOM_TILE_DICT['HEALING']
    return matrix

def remove_trapped_enemies(room):
    '''Remove enemies that are completely encased by walls to prevent unfair spawns, can be used as a post process after all constraints are applied.'''
    matrix = room.room_map
    entities = room.entities

    H = len(matrix)
    W = len(matrix[0])
    count = 0

    for y in range(1, H-1):
        for x in range(1, W-1):
            if matrix[y][x] == ROOM_TILE_DICT['ENEMY']:
                neighbors = [
                    matrix[y+1][x], matrix[y-1][x],
                    matrix[y][x+1], matrix[y][x-1]
                ]
                if all(n == ROOM_TILE_DICT['WALL'] for n in neighbors):
                    count += 1
                    matrix[y][x] = ROOM_TILE_DICT['FLOOR']
                    # Also remove from entities list
                    entities = [e for e in entities if not (isinstance(e, Enemy) and e.x == x and e.y == y)]

    print(f"Saved {count} trapped enemies.")
    return matrix, entities

def clean_generated_doors(matrix, original_matrix):
    height = len(matrix)
    width = len(matrix[0])
    count = 0

    for y in range(height):
        for x in range(width):

            # If it's a door in generated output
            if matrix[y][x] == ROOM_TILE_DICT['DOOR']:

                # Keep ONLY if it was originally a door
                if original_matrix[y][x] != ROOM_TILE_DICT['DOOR']:
                    matrix[y][x] = ROOM_TILE_DICT['FLOOR']
                    count += 1
    print(f"{count} Doors cleaned from generated output.")
    return matrix

def enforce_entity_limits(room, room_type):

    matrix = room.room_map
    entities = room.entities
    limits = {
        "enemy": {3: (24, 48)},
        "loot": {4: (1, 3)},
        "healing": {5: (1, 2)},
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
                matrix[y][x] = ROOM_TILE_DICT['FLOOR']
                room.remove_entity(room.get_entity_at(x, y))
                count -= 1
                removed_count += 1


        # Hard enforce min limits if we are under the minimum limit by placing entities randomly until we reach the minimum limit
        elif count < min_limit:
            # Get all valid floor positions for potential entity placement, excluding edges to prevent unfair placements
            floor_positions = [
                (y, x) for y, x in np.argwhere(matrix == ROOM_TILE_DICT['FLOOR'])
                if 0 < y < matrix.shape[0]-1 and 0 < x < matrix.shape[1]-1
            ]

            np.random.shuffle(floor_positions)  # Shuffle to add randomness to placement

            for y, x in floor_positions[:(min_limit - count)]:
                matrix[y][x] = tile_type
                if tile_type == ROOM_TILE_DICT['ENEMY']:
                    entities.append(Enemy(x, y, None))
                elif tile_type == ROOM_TILE_DICT['CHEST']:
                    entities.append(Chest(x, y, None))
                elif tile_type == ROOM_TILE_DICT['HEALING']:
                    entities.append(HealingFountain(x, y, None))
                added_count += 1


    print(f"Initially had {count} entities in {room_type} room.")
    print(f"Added {added_count} entities and removed {removed_count} entities.")
    print(f"Number of entities in the room after enforcement: {len(entities)}")
    return matrix, entities


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
        # room = apply_matrix_to_room_tiles(room, room.room_map)
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
                room.room_map, room.entities = apply_entities(room, gan_enforced_matrix, mask)

                # Remove walls encasing enemies to prevent unfair spawns
                room.room_map, room.entities = remove_trapped_enemies(room)

                room.room_map, room.entities = enforce_entity_limits(room, room.type)

                # Connectivity Check to make sure all doors have room for movement
                room.room_map = enforce_reachable_door(room)

                # Draw final matrix transform into room tile characters
                # final_room = apply_matrix_to_room_tiles(room, final_matrix)

                return room
            else:
                # final_room = apply_matrix_to_room_tiles(room, gan_matrix)
                return room

        elif model_selection == "diffusion":
            # Generate Diffusion created room
            alphas_cumprod = get_noise_schedule(device=DEVICE)

            diff_tensor = generate_diffusion_dungeon_room(
                GENERATOR,
                ROOM_TYPES[room.type],
                room,
                mask,
                alphas_cumprod,
                DEVICE
            )

            # tensor_to_tilemap
            diff_matrix = diff_tensor.squeeze(0).cpu().numpy() # (H, W)

            unique, counts = np.unique(diff_matrix, return_counts=True)
            print(f"Tensor to Tilemaps Matrix Counts: {dict(zip(unique, counts))}")

            if output_type == "controlled":

                # # enforce structure (walls + doors)
                # for y in range(len(room.room_map)):
                #     for x in range(len(room.room_map[0])):
                #         if mask[y][x] == 1:     # Only modify non-locked tiles
                #             diff_matrix[y][x] = room.room_map[y][x]


                # Post process diffusion output with same constraints as GAN to control room type and ensure playability
                diff_enforced_matrix = enforce_room_type_bias(diff_matrix, room.type)
                room.room_map, room.entities = apply_entities(room, diff_enforced_matrix, mask)
                # room.room_map = clean_generated_doors(diff_matrix, room.room_map)
                room.room_map, room.entities = remove_trapped_enemies(room)
                room.room_map, room.entities = enforce_entity_limits(room, room.type)
                room.room_map = enforce_reachable_door(room)

                # Draw final matrix transform into room tile characters
                # final_room = apply_matrix_to_room_tiles(room, diff_matrix)

                return room
            
            else:
                # final_room = apply_matrix_to_room_tiles(room, diff_matrix)
                return room

    else:
        #Return original room (start or boss)
        # final_room = apply_matrix_to_room_tiles(room, room.room_map)

        return room
