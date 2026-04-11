'''
Create 10K samples of Synthetic Data for GAN training. Mix with handmade data for better training. 
'''
import os
import json
import random
from collections import deque

#User files
from engine.map_generator import Room, assign_room_type, place_doors, extract_room_matrix, enforce_reachable_door
from settings import ROOM_TILE_DICT, ROOM_WIDTH, ROOM_HEIGHT

FLOOR = ROOM_TILE_DICT.get('.')
WALL  = ROOM_TILE_DICT.get('#')
DOOR  = ROOM_TILE_DICT.get('+')
ENEMY = ROOM_TILE_DICT.get('E')
CHEST = ROOM_TILE_DICT.get('C')
HEALING = ROOM_TILE_DICT.get('H')

DATA_PATH = 'game/data/synthetic_rooms_dataset.json'


def ensure_connected(room, doors):
    height = len(room)
    width = len(room[0])

    def neighbors(x, y):
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            yield x + dx, y + dy

    # Fill from first door
    start = doors[0]
    visited = set()
    queue = deque([start])

    while queue:
        x, y = queue.popleft()
        if (x, y) in visited:
            continue
        visited.add((x, y))

        for nx, ny in neighbors(x, y):
            if 0 <= nx < width and 0 <= ny < height:
                if room[ny][nx] in (FLOOR, DOOR):
                    queue.append((nx, ny))

    # Convert unreachable FLOOR into WALL
    for y in range(height):
        for x in range(width):
            if room[y][x] == FLOOR and (x, y) not in visited:
                room[y][x] = WALL

    return room


def enforce_walls_on_edges(room_matrix):
    height = len(room_matrix)
    width = len(room_matrix[0])

    for y in range(height):
        for x in range(width):

            is_edge = (
                x == 0 or x == width - 1 or
                y == 0 or y == height - 1
            )

            if not is_edge:
                continue

            # Keep doors
            if room_matrix[y][x] == DOOR:
                continue

            room_matrix[y][x] = WALL

    return room_matrix


def clear_door_paths(room_matrix, doors):
    for (x, y) in doors:
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy

            if 0 <= nx < len(room_matrix[0]) and 0 <= ny < len(room_matrix):
                if room_matrix[ny][nx] == WALL:
                    room_matrix[ny][nx] = FLOOR


def room_fill(room_matrix, room_type):
    height = len(room_matrix)
    width = len(room_matrix[0])

    # -------------------------------------------------
    # Fill everything with FLOOR (except doors)
    # -------------------------------------------------
    for y in range(height):
        for x in range(width):
            if room_matrix[y][x] != DOOR:
                room_matrix[y][x] = FLOOR

    # -------------------------------------------------
    # Create wall edged structure
    # -------------------------------------------------
    room_matrix = enforce_walls_on_edges(room_matrix)

    # -------------------------------------------------
    # Get door positions
    # -------------------------------------------------
    doors = [(x, y) for y in range(height) for x in range(width) if room_matrix[y][x] == DOOR]

    # -------------------------------------------------
    # Add obstacle clusters
    # -------------------------------------------------
    num_clusters = random.randint(10, 24)

    for _ in range(num_clusters):
        cx = random.randint(1, width - 2)
        cy = random.randint(1, height - 2)

        cluster_size = random.randint(3, 8)

        for _ in range(cluster_size):
            dx = random.randint(-4, 4)
            dy = random.randint(-4, 4)

            nx, ny = cx + dx, cy + dy

            if 0 <= nx < width and 0 <= ny < height:

                # Don't overwrite doors
                if room_matrix[ny][nx] == DOOR:
                    continue

                # Keep space around doors clear
                if any(abs(nx - dx_) + abs(ny - dy_) <= 2 for dx_, dy_ in doors):
                    continue

                room_matrix[ny][nx] = WALL

    # -------------------------------------------------
    # STEP 4: Ensure connectivity (flood fill fix)
    # -------------------------------------------------
    room_matrix = ensure_connected(room_matrix, doors)

    # -------------------------------------------------
    # STEP 5: Place enemies
    # -------------------------------------------------
    if room_type not in ["start", "boss"]:
        enemy_count = random.randint(24, 48)

        placed = 0
        attempts = 0

        while placed < enemy_count and attempts < 100:
            x = random.randint(1, width - 2)
            y = random.randint(1, height - 2)

            if room_matrix[y][x] != FLOOR:
                attempts += 1
                continue

            # Avoid doors
            if any(abs(x - dx_) + abs(y - dy_) <= 3 for dx_, dy_ in doors):
                attempts += 1
                continue

            room_matrix[y][x] = ENEMY
            placed += 1

    return room_matrix



def generate_training_room():
    room = Room(0, 0, ROOM_WIDTH, ROOM_HEIGHT)

    assign_room_type(room)
    place_doors(room.room_map)

    room_matrix = extract_room_matrix(room)

    # Use procedural logic here instead of GAN
    room_matrix = room_fill(room_matrix, room.type)

    room_matrix = enforce_reachable_door(room_matrix)

    return room_matrix, room.type


def save_dataset(dataset, path=DATA_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(dataset, f)

    print(f"\nSynthetic Dataset saved to {path}")



