import math
import pygame

#User files
from engine.entity_system import Enemy, Player
from engine.game_state_system import GameStateSystem
from engine.weapon_factory import Weapon
from utils.sprite_sheet_selection import SpriteSheet
from settings import  DIRECTION_VECTORS, FACE_COLS, HAND_OFFSETS, HANDLE_POSITIONS, ROOM_TILE_DICT, SCREEN_WIDTH, SCREEN_HEIGHT, TILE_SIZE, FPS, ROOM_WIDTH, ROOM_HEIGHT, WEAPON_FLIPPED
from utils.load_and_scale import get_img_hitbox_mask, load_and_scale, load_img
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
bat_grey_img_hit = get_img_hitbox_mask(bat_grey_img)

# #Load loot assets
# chest_1_img = load_and_scale('game/assets/DO Terrain/Terrain/L2_Chest01.PNG', namehint='chest_1')
# chest_1_img.set_colorkey((255, 0, 255, 255))

# #Load healing assets
# fountain_img = load_and_scale('game/assets/DO Terrain/Terrain/L2_Fountain01.PNG', namehint='fountain')
# fountain_img.set_colorkey((255, 0, 255, 255))


#Optional Transparency
# floor_img.set_colorkey((0, 0, 0))

# Load player sprite asset
player_img = load_img("game/assets/players/Males/M_06.png", "player")

# Get Player frame from player sprite sheet
# 64px by 51px with 4 by 3 frames
player_matrix_size = (16, 17) #Size of each frame in the sprite sheet
player_frame_width = 16
player_frame_height = 17
player_col = FACE_COLS["down"]   #Start facing down

player_sheet = SpriteSheet(player_img, player_matrix_size, player_frame_width, player_frame_height)
player_surface = player_sheet.get_sprite_sheet_frame(0, player_col)

player_surface_hit = get_img_hitbox_mask(player_surface)

#Load door sprite asset 
door_img = load_img('game/assets/doors.png', 'doors')
door_img.set_colorkey((255, 255, 255, 0))

# Get door frame from door sprite sheet
# 512px by 512px with 8 by 8 frames
door_matrix_size = (64, 64) #Size of each frame in the sprite sheet
door_frame_width = 32
door_frame_height = 48

door_sheet = SpriteSheet(door_img, door_matrix_size, door_frame_width, door_frame_height)
door_surface = door_sheet.get_sprite_sheet_frame(4, 0, offset=(17,0))   #Move the frame to the right by 17 pixels to get past the white space on the left of the sprite sheet


# Load weapon item assets
# 32px x 32px
sword_1_img = load_img("game/assets/DO Items/Items/Sword01.PNG", "sword_1")
sword_1_img.set_colorkey((255, 0, 255, 255))

pause_overlay = pygame.Surface((ROOM_WIDTH * TILE_SIZE, ROOM_HEIGHT * TILE_SIZE))
pause_overlay.set_alpha(120)  # Set transparency level (0-255)
pause_overlay.fill((0, 0, 0)) # Fill with semi-transparent black

clock = pygame.time.Clock()
running = True
dt = 0

#Create world's starting zone 
world_map = {}

#Create first room on app start
room = generate_dungeon_room()

#Put first room into world map
world_map[room.room_pos] = {
    "room": room,
    "type": room.type,
    "cleared": True     #First room is cleared
}

#Place the player in the center of the first room
center_x, center_y = room.center()
player = Player(center_x, center_y, player_surface)
room.entities.append(player)
player.weapon = room.items[0]    #Give player a starting weapon made in the room for testing, we will update this to make a 3 weapon pick system in starting room.

def draw_room(screen, room):
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
            elif (tile == ROOM_TILE_DICT['FLOOR']
                  or tile == ROOM_TILE_DICT['ENEMY']
                  or tile == ROOM_TILE_DICT['CHEST']
                  or tile == ROOM_TILE_DICT['HEALING']
                  or tile == ROOM_TILE_DICT['PLAYER']
            ):
                #Draws the png for the floors
                screen.blit(floor_img, (x * TILE_SIZE, y * TILE_SIZE))
            elif tile == ROOM_TILE_DICT['DOOR']:
                screen.blit(door_surface, (x * TILE_SIZE, y * TILE_SIZE))

    # -------------
    # Draw entities in the room (player, enemies, chests, healing fountains, etc.)
    # -------------
    for entity in room.entities:
        if isinstance(entity, Player):
            entity.render(screen, player_surface_hit if entity.is_hit else player_surface)
        elif isinstance(entity, Enemy):
            entity.render(screen, bat_grey_img_hit if entity.is_hit else bat_grey_img)
        else:
            entity.render(screen)

    # --------------
    # Draw Player health bar above their head
    # --------------
    player_rect = player.sprite.get_rect(topleft=(player.x * TILE_SIZE, player.y * TILE_SIZE))

    pygame.draw.rect(screen, 'black', (player_rect.x, player_rect.y - 8, TILE_SIZE, 5))   #Always max hp bar background

    current_width = (player.hp / player.max_hp) * TILE_SIZE 

    pygame.draw.rect(screen, 'red', (player_rect.x, player_rect.y - 8, current_width, 5))

    # -------------------
    # Draw player's weapons on them when they have them
    # -------------------
    if player.weapon != None:           #For testing, we will create weapon classes for better clarity
        sword_1_img_flipped = pygame.transform.flip(sword_1_img, WEAPON_FLIPPED[player.facing][0], WEAPON_FLIPPED[player.facing][1])

        handle_x, handle_y = HANDLE_POSITIONS[player.facing]
        handle_offset_x = handle_x - sword_1_img_flipped.get_width() / 2
        handle_offset_y = handle_y - sword_1_img_flipped.get_height() / 2
        rad = math.radians(-player.weapon.attack_angle)
        rot_x = (
            handle_offset_x * math.cos(rad)
            - handle_offset_y * math.sin(rad)
        )
        rot_y = (
            handle_offset_x * math.sin(rad)
            + handle_offset_y * math.cos(rad)
        )
        hand_x = player.x * TILE_SIZE + HAND_OFFSETS[player.facing][0]
        hand_y = player.y * TILE_SIZE + HAND_OFFSETS[player.facing][1]

        sword_1_img_attack = pygame.transform.rotate(sword_1_img_flipped, player.weapon.attack_angle)   # Rotate the sword image for an attack animation effect
        sword_1_img_attack.set_colorkey((255, 0, 255, 255))
        sword_1_img_attack_rect = sword_1_img_attack.get_rect(center=(hand_x - rot_x, hand_y - rot_y))   # Position the attack animation on the correct side of the player based on their facing direction

        screen.blit(sword_1_img_attack, sword_1_img_attack_rect)

game_state_system = GameStateSystem()
while game_state_system.state != "quit":
    #Handle Inputs 
    for event in pygame.event.get():
        game_state_system.update(event)

    #Update Section
    if game_state_system.state == "playing":

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
            elif keys[pygame.K_s]:
                dy = 1
            elif keys[pygame.K_a]:
                dx = -1
            elif keys[pygame.K_d]:
                dx = 1

            if (dx, dy) != (0, 0):
                room, world_map = player.attempt_move(dx, dy, room, world_map)
                player_surface = player_sheet.get_sprite_sheet_frame(row=player.animation_frame, col=FACE_COLS[player.facing])

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
            weapon_damage = player.weapon.attack()
            if isinstance(target_entity, Enemy):
                if weapon_damage:
                    target_entity.defend(weapon_damage)
                    print(f'Attacked enemy! Enemy HP: {target_entity.hp}')
            

        # -------------
        # Update entities in the room (enemies, chests, healing fountains, etc.)
        # -------------
        for entity in room.entities:
            if isinstance(entity, Player):
                entity.update(room)
            if isinstance(entity, Enemy):
                entity.update(player, room)
        for item in room.items:
            if isinstance(item, Weapon):
                item.update()

        #limits FPS to 60
        dt = clock.tick(FPS) / 1000

    #Draw Section we always want this on to keep the screen updated with the current game state, even if paused.
    #We want to keep the screen with existing content while paused instead of filling it with black, so we can render a pause overlay on top of it.
    draw_room(screen, room)

    # If the game is paused, we still want to listen for events (like unpausing or quitting), but we won't update the game state or render the game world. 
    # Instead, we can render a pause overlay or menu.
    if game_state_system.state == "paused":
        # TODO: Add pause menu options and navigation here (resume, settings, quit, etc.)
        paused_background = screen.copy()
        screen.blits((
            (paused_background, (0, 0)), 
            (pause_overlay, (0, 0))
            )
        )

    # Always update the display at the end of the game loop
    pygame.display.flip()
pygame.quit()