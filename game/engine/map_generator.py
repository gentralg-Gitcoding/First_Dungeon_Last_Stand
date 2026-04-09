import random
from collections import deque
from engine.gan_generator import *
from settings import *

# Legend
TILE_DICT = {
    "#": 0,   # wall
    ".": 1,   # floor
    "+": 2,   # door
    "E": 3,   # enemy
    "C": 4,   # chest
    "H": 5    # healing
}

# Room Tracker
ROOMS = []

class Room:
    #Starting point: x, y
    #Area lengths: width and heights w, h
    def __init__(self, x, y, w, h):
        self.x = x     
        self.y = y
        self.w = w
        self.h = h
        self.type = None

    def center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)

    def intersects(self, other):
        return (
            self.x < other.x + other.w and
            self.x + self.w > other.x and
            self.y < other.y + other.h and
            self.y + self.h > other.y
        )

def create_empty_room(width, height):
    return [["#" for _ in range(width)] for _ in range(height)]


def carve_room(game_map, room):
    for y in range(1, room.h - 1):
        for x in range(1, room.w - 1):
            game_map[y][x] = "."


# def carve_h_corridor(game_map, x1, x2, y):
#     for x in range(min(x1, x2), max(x1, x2) + 1):
#         game_map[y][x] = "."


# def carve_v_corridor(game_map, y1, y2, x):
#     for y in range(min(y1, y2), max(y1, y2) + 1):
#         game_map[y][x] = "."

def add_doors(game_map, rooms):
    height = len(game_map)
    width = len(game_map[0])

    for room in rooms:
        for y in range(room.h):
            for x in range(room.w):

                # Only check edge tiles of room
                if (
                    x == room.x or
                    x == room.w or
                    y == room.y or
                    y == room.h
                ):
                    # If this tile is floor and touches corridor
                    if game_map[y][x] == ".":
                        neighbors = [
                            game_map[y-1][x] if y > 0 else "#",
                            game_map[y+1][x] if y < height-1 else "#",
                            game_map[y][x-1] if x > 0 else "#",
                            game_map[y][x+1] if x < width-1 else "#",
                        ]

                        # Corridor touching room edge → place door
                        if neighbors.count(".") >= 2:
                            game_map[y][x] = "+"

def place_doors(room_matrix):
    height = len(room_matrix)
    width = len(room_matrix[0])
    doors = []

    # Top
    room_matrix[0][width // 2] = 2
    doors.append(("top", width // 2, 0))

    # Bottom
    room_matrix[height - 1][width // 2] = 2
    doors.append(("bottom", width // 2, height - 1))

    # Left
    room_matrix[height // 2][0] = 2
    doors.append(("left", 0, height // 2))

    # Right
    room_matrix[height // 2][width - 1] = 2
    doors.append(("right", width - 1, height // 2))

    return doors

def assign_room_types(rooms):
    for i, room in enumerate(rooms):
        if i == 0:
            room.type = "start"
        elif i == len(rooms) - 1:
            room.type = "boss"
        else:
            r = random.random()     #Weight for room type

            if r < 0.1:
                room.type = "empty"
            elif r < 0.2:
                room.type = "loot"
            else:
                room.type = "enemy"

def assign_room_type(room):
    ROOMS.append(room)

    for i, room in enumerate(ROOMS):
        #Check if room already was assigned, skip it
        if not room.type:
            continue

        if i == 0:
            room.type = "start"
        elif i == len(ROOMS) - 1:
            room.type = "boss"
        else:
            r = random.random()     #Weight for room type

            if r < 0.1:
                room.type = "empty"
            elif r < 0.2:
                room.type = "loot"
            else:
                room.type = "enemy"


def extract_room_matrix(game_map, room):
    matrix = []

    for y in range(room.y, room.h):
        row = []
        for x in range(room.x, room.w):
            tile = game_map[y][x]
            row.append(TILE_DICT.get(tile, 0))
        matrix.append(row)

    return matrix

def apply_room_matrix(game_map, room, matrix):
    for y in range(room.h):
        for x in range(room.w):
            value = matrix[y][x]

            for key, val in TILE_DICT.items():
                if val == value:
                    game_map[y][x] = key

def mock_gan_generate(room_matrix, room_type):
    height = len(room_matrix)
    width = len(room_matrix[0])

    new_matrix = [row[:] for row in room_matrix]

    for y in range(1, height - 1):      
        for x in range(1, width - 1):

            #Target only floor tiles, protect walls
            if room_matrix[y][x] != 1:
                continue

            if room_type == "enemy":
                if random.random() < 0.3:
                    new_matrix[y][x] = 3  # enemy

            elif room_type == "loot":
                if random.random() < 0.05:
                    new_matrix[y][x] = 4  # chest

            elif room_type == "empty":
                # if random.random() < 0.05:
                    # new_matrix[y][x] = 5  # healing
                new_matrix[height // 2][width // 2] = 5  # healing

    return new_matrix

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

def apply_mask(original, generated, mask):
    '''
    Cycle through tiles and find safe tiles for GANs to safely modify any changes made. 
    '''
    height = len(original)
    width = len(original[0])

    final = [row[:] for row in original]

    for y in range(height):
        for x in range(width):

            if mask[y][x] == 0:     # 0 masked as safe, give it to GANs
                final[y][x] = generated[y][x]

    return final

def enforce_tile_rules(matrix):
    '''
    Enforce tile changes by ensuring enemies, chests, and healing aren't overlapping and only spawn on floors
    '''
    height = len(matrix)
    width = len(matrix[0])

    for y in range(height):
        for x in range(width):

            if matrix[y][x] in [3, 4, 5]:  # enemy, chest, healing

                # Must be surrounded by floor
                if matrix[y][x] == 3:  # enemy
                    continue

                # If placed on invalid tile → revert to floor
                if matrix[y][x] not in [1, 3, 4, 5]:
                    matrix[y][x] = 1

    return matrix

def is_room_connected(matrix):
    '''
    Create a Breadth-First Search(BFS) algorithm to search throughout the room and ensure there's a way out of it through all doors.
    '''
    height = len(matrix)
    width = len(matrix[0])

    visited = set()
    queue = deque()

    # find first floor tile in the room to begin search
    for y in range(height):
        for x in range(width):
            if matrix[y][x] == 1:
                queue.append((x, y))
                visited.add((x, y))
                break
        if queue:
            break

    while queue:
        x, y = queue.popleft()

        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:       #Check all 4 directions from standing point
            nx = x + dx
            ny = y + dy

            if (
                0 <= nx < width and
                0 <= ny < height and
                (nx, ny) not in visited and
                matrix[ny][nx] in [1, 2]  # floor or door
            ):
                visited.add((nx, ny))
                queue.append((nx, ny))

    # Check all doors reachable
    for y in range(height):
        for x in range(width):
            if matrix[y][x] == 2 and (x, y) not in visited:
                return False

    return True 

# def generate_dungeon(width = ROOM_WIDTH, height = ROOM_HEIGHT, max_rooms=10):
#     rooms = []

#     #Fills entire dungeon with walls
#     game_map = create_empty_map(width, height)

#     #Build rooms
#     for _ in range(max_rooms):
#         # w = random.randint(4, 8)
#         # h = random.randint(4, 8)

#         # x = random.randint(1, width - w - 1)
#         # y = random.randint(1, height - h - 1)

#         # new_room = Room(x, y, w, h)
        
#         new_room = Room(0, 0, width, height)

#         # if any(new_room.intersects(other) for other in rooms):
#         #     continue

#         # Build room floors
#         carve_room(game_map, new_room)

#         # if rooms:
#         #     prev_x, prev_y = rooms[-1].center()
#         #     new_x, new_y = new_room.center()

#         #     if random.random() < 0.5:
#         #         carve_h_corridor(game_map, prev_x, new_x, prev_y)
#         #         carve_v_corridor(game_map, prev_y, new_y, new_x)
#         #     else:
#         #         carve_v_corridor(game_map, prev_y, new_y, prev_x)
#         #         carve_h_corridor(game_map, prev_x, new_x, new_y)

#         rooms.append(new_room)

#     # add_doors(game_map, rooms)

#     assign_room_types(rooms)

#     for room in rooms:
#         # Get room matrix on the map
#         base_matrix = extract_room_matrix(game_map, room)

#         place_doors(base_matrix)

#         # new_matrix = mock_gan_generate(matrix, room.type)

#         # apply_room_matrix(game_map, room, new_matrix)

#         # Generate GAN created room
#         gan_matrix = generate_room(Generator(), room.w, room.h)

#         # Create structure mask
#         mask = create_structure_mask(base_matrix)

#         # Apply constraints to control the GAN
#         final_matrix = apply_mask(base_matrix, gan_matrix, mask)
#         final_matrix = enforce_tile_rules(final_matrix)

#         # Connectivity Check
#         if not is_room_connected(final_matrix):
#             final_matrix = base_matrix  # fallback if room cant be connected

#         # Draw final matrix to map
#         apply_room_matrix(game_map, room, final_matrix)

#     return game_map, rooms


def generate_dungeon_room(width = ROOM_WIDTH, height = ROOM_HEIGHT):
    room = Room(0, 0, width, height)

    #Fill room with walls
    room_map = create_empty_room(room.w, room.h)

    # Build room floors
    carve_room(room_map, room)

    #Assign the type of room
    assign_room_type(room)

    # Get room matrix on the map
    base_matrix = extract_room_matrix(room_map, room)

    place_doors(base_matrix)

    # Generate GAN created room
    gan_matrix = generate_room(Generator(), room.w, room.h)

    # Create structure mask
    mask = create_structure_mask(base_matrix)

    # Apply constraints to control the GAN
    final_matrix = apply_mask(base_matrix, gan_matrix, mask)
    final_matrix = enforce_tile_rules(final_matrix)

    # Connectivity Check
    # if not is_room_connected(final_matrix):
    #     final_matrix = base_matrix  # fallback if room cant be connected

    # Draw final matrix to map
    apply_room_matrix(room_map, room, final_matrix)

    return room_map, room