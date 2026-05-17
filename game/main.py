import pygame

#User files
from engine.entity_system import Enemy, Player
from engine.game_state_system import GameStateSystem
from utils.sprite_sheet_selection import get_img_frame_surface
from settings import  DIRECTION_VECTORS, ROOM_TILE_DICT, SCREEN_WIDTH, SCREEN_HEIGHT, TILE_SIZE, FPS, ROOM_WIDTH, ROOM_HEIGHT
from utils.load_and_scale import load_and_scale, load_img
from engine.map_generator import generate_dungeon_room


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


# TODO: Create a function that can take in any sprite sheet and frame dimensions to return the correct frame as a surface. Currently hardcoded for player and door sprite sheets.
# Possibly use a class to represent sprite sheets and their frames for better organization and reusability.
# Get Player frame from player sprite sheet
# 64px by 51px with 4 by 3 frames
player_frame_width = 16
player_frame_height = 17
player_frame = pygame.Rect(
    0 * player_frame_width,     #0, 0 is top-left frame
    0 * player_frame_height, 
    player_frame_width, 
    player_frame_height
)     

#Resize the character to be a tile size
player_surface = get_img_frame_surface(player_img, player_frame)

#Load door sprite asset 
door_img = load_img('game/assets/doors.png', 'doors')
door_img.set_colorkey((255, 255, 255, 0))

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
door_surface = get_img_frame_surface(door_img, door_frame)

pause_overlay = pygame.Surface((ROOM_WIDTH * TILE_SIZE, ROOM_HEIGHT * TILE_SIZE))
pause_overlay.set_alpha(120)  # Set transparency level (0-255)
pause_overlay.fill((0, 0, 0)) # Fill with semi-transparent black

clock = pygame.time.Clock()
running = True
dt = 0

#Create world's starting zone 
world_map = {}
room_pos = (0, 0)

#Create first room on app start
room = generate_dungeon_room()

#Put first room into world map
world_map[room_pos] = {
    "room": room,
    "type": room.type,
    "cleared": True     #First room is cleared
}

#Place the player in the center of the first room
center_x, center_y = room.center()
player = Player(center_x, center_y, player_img)
room.entities.append(player)

def get_direction(dx, dy):
    '''Get the direction the player moves in'''
    if(dx == 1 and dy == 0):
        return 'right' 

    elif(dx == 0 and dy == -1):
        return  'top'

    elif(dx == -1 and dy == 0):
        return 'left'

    elif(dx == 0 and dy == 1):
        return 'bottom'

def set_player_position(direction):
    '''Reposition player depending on door direction you move towards when transitioning rooms'''
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

def move_rooms(room_pos, direction):
    '''Updates the room in the world map the player moved to'''
    x, y = room_pos

    if direction == 'right':
        return (x + 1, y)
    if direction == 'top':
        return (x, y - 1)
    if direction == 'left':
        return (x - 1, y)
    if direction == 'bottom':
        return (x, y + 1)

def check_door_transition(target_x, target_y, room_map=room.room_map):
    '''Checks if player is on a door tile to transition rooms'''
    px, py = target_x, target_y

    if room_map[py][px] == ROOM_TILE_DICT['DOOR']:
        return True

    return False

def handle_room_transition(player_position, transition_direction, room_pos, room=room):
    '''Handles player transitioning between rooms when stepping on a door tile'''
    # Delay player movement for a short time to prevent multiple room transitions from one key press due to the player still being on the door tile for multiple frames. 
    # This is a temporary solution until we implement seamless movement and better input handling.
    player.transition_cooldown = pygame.time.get_ticks() + 250


    if not transition_direction:
        return player_position, room_pos, room

    new_pos = move_rooms(room_pos, transition_direction)

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
    player_position = set_player_position(transition_direction)

    return player_position, room_pos, room 

def attempt_move(entity, dx, dy, room, room_pos):
    '''Movement check for player movement and room transitions. 
    Checks if player is trying to move onto a door tile to transition rooms, or if the tile they are trying to move onto is blocked.'''
    target_x = entity.x + dx
    target_y = entity.y + dy

    #Check which way the player went
    transition_direction = get_direction(dx, dy)

    #Check if target is a door and if so, handle room transition
    if room.room_map[target_y][target_x] == ROOM_TILE_DICT['DOOR']:
        (entity.x, entity.y), room_pos, room = handle_room_transition(entity.position, transition_direction, room_pos, room)
        return entity.x, entity.y, room_pos, room
    elif not room.is_blocked(target_x, target_y):
        entity.move(dx, dy)
        return entity.x, entity.y, room_pos, room
    else:
        return entity.x, entity.y, room_pos, room

def draw_room(screen, room):
    # '''Draws the room's tiles and entities'''
    # for y in range(ROOM_HEIGHT):
    #     for x in range(ROOM_WIDTH):
    #         tile = room.room_map[y][x]
    #         if tile == ROOM_TILE_DICT['FLOOR']:
    #             screen.blit(floor_img, (x * TILE_SIZE, y * TILE_SIZE))
    #         elif tile == ROOM_TILE_DICT['WALL']:
    #             screen.blit(wall_img, (x * TILE_SIZE, y * TILE_SIZE))
    #         elif tile == ROOM_TILE_DICT['DOOR']:
    #             screen.blit(door_surface, (x * TILE_SIZE, y * TILE_SIZE))

    # for entity in room.entities:
    #     entity.draw(screen)


    # fill the screen to wipe away anything from last frame
    screen.fill("black")

    # -------------
    # Draw Dungeon Tiles
    # -------------
    for y, row in enumerate(room.room_map):
        for x, tile in enumerate(row):
            if tile == ROOM_TILE_DICT['WALL']:
                #Draws the png for the walls
                screen.blit(wall_img, (x * TILE_SIZE, y * TILE_SIZE))
            elif tile == ROOM_TILE_DICT['FLOOR']:
                #Draws the png for the floors
                screen.blit(floor_img, (x * TILE_SIZE, y * TILE_SIZE))
            elif tile == ROOM_TILE_DICT['DOOR']:
                screen.blit(door_surface, (x * TILE_SIZE, y * TILE_SIZE))
            # TODO: Refactor to use Entity system instead of hardcoding tile types here. 
            # Would allow for more dynamic interactions and behaviors for different tile types (enemies, loot, healing, etc.) instead of just rendering a static image.
            elif tile == ROOM_TILE_DICT['ENEMY']:
                screen.blits((
                    (floor_img, (x * TILE_SIZE, y * TILE_SIZE)),
                    (bat_grey_img, (x * TILE_SIZE, y * TILE_SIZE))
                    )
                )
            elif tile == ROOM_TILE_DICT['CHEST']:
                screen.blits((
                    (floor_img, (x * TILE_SIZE, y * TILE_SIZE)),
                    (chest_1_img, (x * TILE_SIZE, y * TILE_SIZE))
                    )
                )
            elif tile == ROOM_TILE_DICT['HEALING']:
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

    #Draw Player tile
    screen.blit(player_surface, (player.x * TILE_SIZE, player.y * TILE_SIZE))


game_state_system = GameStateSystem()
while game_state_system.state != "quit":
    for event in pygame.event.get():
        game_state_system.update(event)

    #Update Section
    if game_state_system.state == "playing":
        #Handle Inputs 
        # for event in pygame.event.get():
            # game_state_system.update(event)

        # ------------------------------------------
        # Continous movement handling for when keys are held down
        # -------------------------------------------
        current_time = pygame.time.get_ticks()
        dx = 0
        dy = 0

        if current_time - player.last_move_time > player.move_delay and current_time > player.transition_cooldown:
            keys = pygame.key.get_pressed()

            if keys[pygame.K_w]:
                dy = -1
                player.facing = "up"
            elif keys[pygame.K_s]:
                dy = 1
                player.facing = "down"
            elif keys[pygame.K_a]:
                dx = -1
                player.facing = "left"
            elif keys[pygame.K_d]:
                dx = 1
                player.facing = "right"

            if (dx, dy) != (0, 0):
                player.x, player.y, room_pos, room = attempt_move(player, dx, dy, room, room_pos)
                player.last_move_time = current_time


        # ------------------------------------------
        # Interaction handling (E key)
        # ------------------------------------------
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e:
                facing_dx, facing_dy = DIRECTION_VECTORS[player.facing]

                facing_target_x = player.x + facing_dx
                facing_target_y = player.y + facing_dy

                target_entity = room.get_entity_at(facing_target_x, facing_target_y)

                if target_entity:
                    target_entity.interact(player, room)


        # ------------------------------------------
        # Combat handling (MouseButton Down key)
        # ------------------------------------------
        if event.type == pygame.MOUSEBUTTONDOWN:
            facing_dx, facing_dy = DIRECTION_VECTORS[player.facing]

            facing_target_x = player.x + facing_dx
            facing_target_y = player.y + facing_dy

            target_entity = room.get_entity_at(facing_target_x, facing_target_y)
            if current_time - player.last_attack_time >= player.attack_speed:
                player.last_attack_time = current_time
                if isinstance(target_entity, Enemy):
                    target_entity.hp -= player.attack
                    print(f'Attacked enemy! Enemy HP: {target_entity.hp}')

                    if target_entity.hp <= 0:
                        print('Enemy defeated!')
                        room.remove_entity(target_entity)


        

        # -------------
        # Update entities in the room (enemies, chests, healing fountains, etc.)
        # -------------
        for entity in room.entities:
            if isinstance(entity, Enemy):
                entity.update(player, room)


        #Updates the full display surface to the screen
        # pygame.display.flip()

        #limits FPS to 60
        # clock.tick(FPS)
        dt = clock.tick(FPS) / 1000


    #Draw Section
    #We want to keep the screen with existing content while paused instead of filling it with black, so we can render a pause overlay on top of it.
    draw_room(screen, room)


    # If the game is paused, we still want to listen for events (like unpausing or quitting), but we won't update the game state or render the game world. 
    # Instead, we can render a pause overlay or menu.
    if game_state_system.state == "paused":
        # TODO: Add pause menu options and navigation here (resume, settings, quit, etc.)
        screen.blit(pause_overlay, (0, 0))

    pygame.display.flip()
pygame.quit()