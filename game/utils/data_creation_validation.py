from collections import deque

#User files
from settings import ROOM_TILE_DICT


DOOR = ROOM_TILE_DICT['DOOR']
FLOOR = ROOM_TILE_DICT['FLOOR']
ENEMY = ROOM_TILE_DICT['ENEMY']
CHEST = ROOM_TILE_DICT['CHEST']
HEALING = ROOM_TILE_DICT['HEALING']

WALKABLE_TILES = {FLOOR, ENEMY, CHEST, HEALING}

def doors_reachable(room):
    '''
    Use the Breadth-First Search algorithm to search all vertices of the room
    '''
    height = len(room)
    width = len(room[0])

    #Get all door coordinates
    doors = [(x, y) for y in range(height) for x in range(width) if room[y][x] == DOOR]

    if len(doors) <= 1:
        return True  # trivial case

    visited = set()
    queue = deque([doors[0]])

    #Get each location next to tile
    def neighbors(x, y):
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                yield nx, ny

    while queue:
        x, y = queue.popleft()
        if (x, y) in visited:
            continue
        visited.add((x, y))

        for nx, ny in neighbors(x, y):
            if room[ny][nx] in WALKABLE_TILES or room[ny][nx] == DOOR:
                queue.append((nx, ny))

    # Check all doors visited
    return all((x, y) in visited for (x, y) in doors)


def no_blocked_doors(room):
    height = len(room)
    width = len(room[0])

    def neighbors(x, y):
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            yield x + dx, y + dy

    for y in range(height):
        for x in range(width):
            if room[y][x] == DOOR:
                has_exit = False

                for nx, ny in neighbors(x, y):
                    if 0 <= nx < width and 0 <= ny < height:
                        if room[ny][nx] == FLOOR:
                            has_exit = True
                            break

                if not has_exit:
                    return False

    return True


def has_enough_floor(room, room_type, min_ratio=0.3, max_ratio=0.9):
    if room_type in ["start", "boss", "healing", "loot"]:
        return True

    height = len(room)
    width = len(room[0])

    total = height * width
    floor_count = sum(
        1 
        for y in range(height)
        for x in range(width)
        if room[y][x] in WALKABLE_TILES
    )

    ratio = floor_count / total

    return min_ratio <= ratio <= max_ratio


def is_valid_room(room, room_type):
    return (
        no_blocked_doors(room)
        and doors_reachable(room)
        and has_enough_floor(room, room_type)
    )