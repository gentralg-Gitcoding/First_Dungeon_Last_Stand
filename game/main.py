import sys
import pygame

#User files
from settings import *
from utils.load_and_scale import *



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

clock = pygame.time.Clock()
running = True
dt = 0


# Legend
#   # = wall
#   . = floor
#   @ = player start

dungeon_map = [
"#########################",
"#.......................#",
"#.......................#",
"#..........@............#",
"#.......................#",
"#.......................#",
"#.......................#",
"#.......................#",
"#.......................#",
"#.......................#",
"#.......................#",
"#.......................#",
"#.......................#",
"#.......................#",
"#.......................#",
"#.......................#",
"#.......................#",
"#########################",
]

#Track the player with iterations through the array
for y, row in enumerate(dungeon_map):
    for x, tile in enumerate(row):
        if tile == "@":
            player_x = x
            player_y = y

# TODO: Create a trackable vector for seamless movements when a key is held down 
# player_pos = pygame.Vector2(player_x * TILE_SIZE, player_y * TILE_SIZE)

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
                0 <= new_x < len(dungeon_map[0]) and
                0 <= new_y < len(dungeon_map) and
                dungeon_map[new_y][new_x] != "#"
            ):
                player_x = new_x
                player_y = new_y

    # keys = pygame.key.get_pressed()
    # if keys[pygame.K_w]:
    #     player_pos.y -= 300 * dt
    # if keys[pygame.K_s]:
    #     player_pos.y += 300 * dt
    # if keys[pygame.K_a]:
    #     player_pos.x -= 300 * dt
    # if keys[pygame.K_d]:
    #     player_pos.x += 300 * dt

    #Setup wall blocking (ORDER MATTERS!)
    # if (
    #     0 <= player_pos.x < len(dungeon_map[0]) and
    #     0 <= player_pos.y < len(dungeon_map) and
    #     dungeon_map[player_pos.y][player_pos.x] != "#"
    # ):
    #     player_x = player_pos.x
    #     player_y = player_pos.y


    #Update Section



    #Draw Section

    # fill the screen to wipe away anything from last frame
    screen.fill("black")

    #Draw Dungeon Tiles
    for y, row in enumerate(dungeon_map):
        for x, tile in enumerate(row):
            if tile == "#":
                #Draws the png for the walls
                screen.blit(wall_img, (x * TILE_SIZE, y * TILE_SIZE), ) 
            elif tile == ".":
                #Draws the png for the floors
                screen.blit(floor_img, (x * TILE_SIZE, y * TILE_SIZE))
            elif tile == "@":
                #Draws the png for the floors
                screen.blit(floor_img, (x * TILE_SIZE, y * TILE_SIZE))

    # fill the player pos with the floor tile to wipe away anything from last frame
    # screen.blit(floor_img, (player_x * TILE_SIZE, player_y * TILE_SIZE))

    # Get Player frame from sprite sheet on sprite img 
    # 64px by 51px with 4 by 3 sprites
    frame_width = 16
    frame_height = 17
    player_frame = pygame.Rect(0, 0, frame_width, frame_height)

    #Pick direction frames
    direction = 0  # 0 = down

    player_frame = pygame.Rect(
        0,                    # column (animation frame)
        direction * frame_height,
        frame_width,
        frame_height
    )

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