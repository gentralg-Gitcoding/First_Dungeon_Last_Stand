import pygame

#User files
from settings import  SCREEN_WIDTH, SCREEN_HEIGHT, TILE_SIZE, FPS, ROOM_WIDTH, ROOM_HEIGHT
from utils.load_and_scale import load_and_scale, load_img
from engine.map_generator import generate_dungeon_room



WALL = 'WALL'
FLOOR = 'FLOOR'
DOOR = 'DOOR'
ENEMY = 'ENEMY'
CHEST = 'CHEST'
HEALING = 'HEALING'

pygame.init()

#Screen Size
screen = pygame.display.set_mode(size=(SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("First Dungeon Last Stand")


# Load and scale wall and floor assets to be 32x32 from utils
floor_img = load_and_scale("game/assets/tiles/Brick_01.png", namehint="floor")
wall_img = load_and_scale("game/assets/tiles/Brickwall5_Texture.png", namehint="wall")

#Load enemy assets
bat_grey_img = load_and_scale('game/assets/DO Monsters/Monsters/BatGrey.PNG', namehint='bat_grey')
bat_grey_img.set_colorkey((255, 0, 255, 255))

#Load loot assets
chest_1_img = load_and_scale('game/assets/DO Terrain/Terrain/L2_Chest01.PNG', namehint='chest_1')
chest_1_img.set_colorkey((255, 0, 255, 255))

#Load healing assets
fountain_img = load_and_scale('game/assets/DO Terrain/Terrain/L2_Fountain01.PNG', namehint='fountain')
fountain_img.set_colorkey((255, 0, 255, 255))


#Optional Transparency
# floor_img.set_colorkey((0, 0, 0))

# Load player sprite asset
player_img = load_img("game/assets/players/Males/M_06.png", "player")

#Load door sprite asset 
door_img = load_img('game/assets/doors.png', 'doors')
door_img.set_colorkey((255, 255, 255, 0))

clock = pygame.time.Clock()
running = True
dt = 0

#Create world's starting zone 
world_map = {}
room_pos = (0, 0)
direction = ''

#Create first room on app start
room = generate_dungeon_room()

#Put first room into world map
world_map[room_pos] = {
    "room": room,
    "type": room.type,
    "cleared": True     #First room is cleared
}

#Place the player in the center of the first room
player_x, player_y = room.center()

#Get the direction of the door the player moves in
def get_direction(px, py):
    if(px == ROOM_WIDTH - 1 and py == ROOM_HEIGHT // 2):
        #Player exits right
        return 'right' 

    elif(px == ROOM_WIDTH // 2 and py == 0):
        #Player exits top
        return  'top'

    elif(px == 0 and py == ROOM_HEIGHT // 2):
        #Player exits left
        return 'left'

    elif(px == ROOM_WIDTH // 2 and py == ROOM_HEIGHT - 1):
        #Player exits bottom
        return 'bottom'


#Reposition player depending on door direction you move to
def set_player_position(direction):
    if(direction == 'right'):
        #Player exits right
        return 1, ROOM_HEIGHT // 2

    elif(direction == 'top'):
        #Player exits top
        return ROOM_WIDTH // 2, ROOM_HEIGHT - 2

    elif(direction == 'left'):
        #Player exits left
        return ROOM_WIDTH - 2, ROOM_HEIGHT // 2

    elif(direction == 'bottom'):
        #Player exits bottom
        return ROOM_WIDTH // 2, 1

#Updates the room in the world map the player moved to
def move_rooms(room_pos, direction):
    x, y = room_pos

    if direction == 'right':
        return (x + 1, y)
    if direction == 'top':
        return (x, y - 1)
    if direction == 'left':
        return (x - 1, y)
    if direction == 'bottom':
        return (x, y + 1)


# TODO: Create a trackable vector for seamless movements when a key is held down 
# player_pos = pygame.Vector2(player_x * TILE_SIZE, player_y * TILE_SIZE)

def check_door_transition(player_x, player_y, room_map=room.room_map):
    px, py = player_x, player_y

    if room_map[py][px] == DOOR:
        return True

    return False

def handle_room_transition():
    global room_pos, direction, room, player_x, player_y

    #Check which way the player went
    direction = get_direction(player_x, player_y)

    if not direction:
        return player_x, player_y

    new_pos = move_rooms(room_pos, direction)

    if new_pos in world_map:
        # Room already exists
        # print(f'Room: {new_pos}')
        # print(world_map[new_pos]['room'].type)
        room = world_map[new_pos]['room']
    else:
        # Generate new room
        new_room = generate_dungeon_room()
        # print(f'Entered new room: {new_pos}')
        # print(f'New room type: {new_room.type}')
        world_map[new_pos] = {
            'room': new_room,
            'type': new_room.type,
            'cleared': False
        }
        room = new_room

    room_pos = new_pos

    # reposition player depending on door used
    player_x, player_y = set_player_position(direction)

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
                0 <= new_x < len(room.room_map[0]) and
                0 <= new_y < len(room.room_map) and
                room.room_map[new_y][new_x] != WALL
            ):
                # print(player_x, player_y)
                player_x = new_x
                player_y = new_y

            #Check if player transitioned rooms
            if check_door_transition(player_x, player_y, room.room_map):

                handle_room_transition()



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

    # -------------
    # Draw Dungeon Tiles
    # -------------
    for y, row in enumerate(room.room_map):
        for x, tile in enumerate(row):
            if tile == WALL:
                #Draws the png for the walls
                screen.blit(wall_img, (x * TILE_SIZE, y * TILE_SIZE))
            elif tile == FLOOR:
                #Draws the png for the floors
                screen.blit(floor_img, (x * TILE_SIZE, y * TILE_SIZE))
            elif tile == DOOR:
                screen.blit(door_surface, (x * TILE_SIZE, y * TILE_SIZE))
            elif tile == ENEMY:
                screen.blits((
                    (floor_img, (x * TILE_SIZE, y * TILE_SIZE)),
                    (bat_grey_img, (x * TILE_SIZE, y * TILE_SIZE))
                    )
                )
            elif tile == CHEST:
                screen.blits((
                    (floor_img, (x * TILE_SIZE, y * TILE_SIZE)),
                    (chest_1_img, (x * TILE_SIZE, y * TILE_SIZE))
                    )
                )
            elif tile == HEALING:
                screen.blits((
                    (floor_img, (x * TILE_SIZE, y * TILE_SIZE)),
                    (fountain_img, (x * TILE_SIZE, y * TILE_SIZE))
                    )
                )


    #VISUAL DEBUG ONLY
    cx, cy = room.center()

    if room.type == "start":
        color = (0, 255, 0)
    elif room.type == "boss":
        color = (255, 0, 0)
    elif room.type == "loot":
        color = (255, 255, 0)
    elif room.type == "healing":
        color = (255, 0, 255)
    else:
        color = (100, 100, 255)

    pygame.draw.circle(
        screen,
        color,
        (cx * TILE_SIZE, cy * TILE_SIZE),
        5
    )

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