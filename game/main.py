import sys
import pygame
import random

#User files
from settings import *
from utils.load_and_scale import *
from engine.map_generator import generate_dungeon_room, ROOMS



pygame.init()

#Screen Size
screen = pygame.display.set_mode(size=(SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("First Dungeon Last Stand")


# Load and scale wall and floor assets to be 32x32 from utils
floor_img = load_and_scale("game/assets/tiles/Brick_01.png", namehint="floor")
wall_img = load_and_scale("game/assets/tiles/Brickwall5_Texture.png", namehint="wall")


#Optional Transparency
# floor_img.set_colorkey((0, 0, 0))

# Load player sprite asset
player_img = load_img("game/assets/players/Males/M_06.png", "player")

#Load door sprite asset 
door_img = load_img('game/assets/doors.png', 'doors')
# print(door_img.get_at([0,0]))
door_img.set_colorkey((255, 255, 255, 0))

clock = pygame.time.Clock()
running = True
dt = 0




# dungeon_map, rooms = generate_dungeon(ROOM_WIDTH, ROOM_HEIGHT)
dungeon_room, room = generate_dungeon_room(ROOM_WIDTH, ROOM_HEIGHT)



#Place the player in the center of first room
player_x, player_y = ROOMS[0].center()

#Player exits right
spawn_x = 1
spawn_y = ROOM_HEIGHT // 2

#Player exits top
spawn_x = ROOM_WIDTH // 2
spawn_y = ROOM_HEIGHT - 2

# TODO: Create a trackable vector for seamless movements when a key is held down 
# player_pos = pygame.Vector2(player_x * TILE_SIZE, player_y * TILE_SIZE)

def check_door_transition(player_x, player_y, room_matrix=dungeon_room):
    px, py = player_x, player_y

    if room_matrix[py][px] == 2:
        return True

    return False

while running:

    #Set Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            new_x = player_x
            new_y = player_y

            #Setup movement keys (W,A,S,D)
            if event.key == pygame.K_w:
                new_y -= 1
            if event.key == pygame.K_s:
                new_y += 1
            if event.key == pygame.K_a:
                new_x -= 1
            if event.key == pygame.K_d:
                new_x += 1

            #Setup wall blocking (ORDER MATTERS!)
            if (
                0 <= new_x < len(dungeon_room[0]) and
                0 <= new_y < len(dungeon_room) and
                dungeon_room[new_y][new_x] != "#"
            ):
                player_x = new_x
                player_y = new_y

            #Check if player transitioned rooms
            if check_door_transition(player_x, player_y, dungeon_room):
                current_room = ROOMS.index(random.randrange(1,9))

                # reposition player depending on door used
                player_x.set_position(spawn_x)
                player_y.set_position(spawn_y)

    # keys = pygame.key.get_pressed()
    # if keys[pygame.K_w]:
    #     player_pos.y -= 300 * dt
    # if keys[pygame.K_s]:
    #     player_pos.y += 300 * dt
    # if keys[pygame.K_a]:
    #     player_pos.x -= 300 * dt
    # if keys[pygame.K_d]:
    #     player_pos.x += 300 * dt


    #Update Section



    #Draw Section

    # fill the screen to wipe away anything from last frame
    screen.fill("black")

    # Get door frame from door sprite sheet
    # 512px by 512px with 8 by 8 frames
    door_frame_width = 64
    door_frame_height = 64
    door_frame = pygame.Rect(
        0 * door_frame_width, 
        4 * door_frame_height, 
        door_frame_width, 
        door_frame_height
    )

    #Resize the door to be a tile size
    door_surface = door_img.subsurface(door_frame).copy()
    door_surface = pygame.transform.scale(door_surface, (TILE_SIZE, TILE_SIZE))


    #Draw Dungeon Tiles
    for y, row in enumerate(dungeon_room):
        for x, tile in enumerate(row):
            if tile == "#":
                #Draws the png for the walls
                screen.blit(wall_img, (x * TILE_SIZE, y * TILE_SIZE))
            elif tile == ".":
                #Draws the png for the floors
                screen.blit(floor_img, (x * TILE_SIZE, y * TILE_SIZE))
            elif tile == "+":
                screen.blit(door_surface, (x * TILE_SIZE, y * TILE_SIZE))


    #VISUAL DEBUG ONLY
    for i, room in enumerate(ROOMS):
        cx, cy = room.center()

        if room.type == "start":
            color = (0, 255, 0)
        elif room.type == "boss":
            color = (255, 0, 0)
        elif room.type == "loot":
            color = (255, 255, 0)
        elif room.type == "empty":
            color = (255, 0, 255)
        else:
            color = (100, 100, 255)

        pygame.draw.circle(
            screen,
            color,
            (cx * TILE_SIZE, cy * TILE_SIZE),
            5
        )
        # print(f'Room: {i}')

    # Get Player frame from player sprite sheet
    # 64px by 51px with 4 by 3 frames
    frame_width = 16
    frame_height = 17
    player_frame = pygame.Rect(0 * frame_width, 0 * frame_height, frame_width, frame_height)     #0, 0 is top-left frame

    #Resize the character to be a tile size
    player_surface = player_img.subsurface(player_frame).copy()
    player_surface = pygame.transform.scale(player_surface, (TILE_SIZE, TILE_SIZE))

    #Draw Player tile
    screen.blit(player_surface, (player_x * TILE_SIZE, player_y * TILE_SIZE))

    #Updates the full display surface to the screen
    pygame.display.flip()

    #limits FPS to 60
    clock.tick(FPS)
    # dt = clock.tick(60) / 1000
pygame.quit()