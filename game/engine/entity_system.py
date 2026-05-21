import pygame

from settings import ROOM_HEIGHT, ROOM_TILE_DICT, ROOM_WIDTH, TILE_SIZE

class Entity:
    def __init__(
        self,
        x,
        y,
        sprite=None,
        blocks_movement=False,
        name="entity"
    ):
        self.x = x
        self.y = y

        self.sprite = sprite
        self.blocks_movement = blocks_movement
        self.name = name

        self.active = True

    @property
    def position(self):
        return (self.x, self.y)

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def update(self, room):
        """
        Called every game tick/frame.
        Override in subclasses.
        """
        pass

    def interact(self, player, room):
        """
        Called when the player interacts with entity.
        Override in subclasses.
        """
        pass

    def render(self, screen, sprite=None, tile_size=TILE_SIZE):
        """
        Draw entity to screen.
        """
        if sprite:
            screen.blit(
                sprite,
                (self.x * tile_size, self.y * tile_size)
            )
        else:
            screen.blit(
                self.sprite,
                (self.x * tile_size, self.y * tile_size)
            )


class Player(Entity):
    def __init__(self, x, y, sprite):
        super().__init__(
            x,
            y,
            sprite=sprite,
            blocks_movement=False,
            name="PLAYER"
        )

        self.max_hp = 100
        self.hp = 100
        self.attack = 10
        self.attack_speed = 500
        self.move_delay = 200
        self.last_move_time = 0
        self.last_attack_time = 0
        self.transition_cooldown = 0
        self.facing = "down"
        self.is_hit = False
        self.last_hit_time = 0
        self.hit_cooldown = 100

    def update(self, room):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_hit_time >= self.hit_cooldown:
            self.is_hit = False
            self.last_hit_time = current_time

    def get_direction(self, dx, dy):
        '''Get the direction the player moves in'''
        if(dx == 1 and dy == 0):
            self.facing = "right"
            return 'right'

        elif(dx == 0 and dy == -1):
            self.facing = "up"
            return 'top'

        elif(dx == -1 and dy == 0):
            self.facing = "left"
            return 'left'

        elif(dx == 0 and dy == 1):
            self.facing = "down"
            return 'bottom'

    def handle_room_transition(self, room, world_map, transition_direction):
        '''Handles player transitioning between rooms when stepping on a door tile'''
        # Delay player movement for a short time to prevent multiple room transitions from one key press due to the player still being on the door tile for multiple frames. 
        # This is a temporary solution until we implement seamless movement and better input handling.
        self.transition_cooldown = pygame.time.get_ticks() + 250

        if not transition_direction:
            return room

        room, world_map = room.move_rooms(transition_direction, world_map)
        room = self.set_player_position(room, transition_direction)
        return room, world_map

    def attempt_move(self, dx, dy, room, world_map):
        '''Movement check for player movement and room transitions. 
        Checks if player is trying to move onto a door tile to transition rooms, or if the tile they are trying to move onto is blocked.'''
        target_x = self.x + dx
        target_y = self.y + dy

        #Check which way the player went
        transition_direction = self.get_direction(dx, dy)

        #Check if target is a door and if so, handle room transition
        if room.room_map[target_y][target_x] == ROOM_TILE_DICT['DOOR']:
            room, world_map = self.handle_room_transition(room, world_map, transition_direction)
            return room, world_map
        elif not room.is_blocked(target_x, target_y):
            room.update_entity_position(self, self.x + dx, self.y + dy)
            self.move(dx, dy)
            return room, world_map
        else:
            return room, world_map

    def set_player_position(self, room, transition_direction):
        '''Reposition player depending on door direction you move towards when transitioning rooms'''
        if(transition_direction == 'right'):
            #Player exits right
            self.x, self.y = 1, ROOM_HEIGHT // 2
            room.entities.append(self)
            return room 

        elif(transition_direction == 'top'):
            #Player exits top
            self.x, self.y = ROOM_WIDTH // 2, ROOM_HEIGHT - 2
            room.entities.append(self)
            return room

        elif(transition_direction == 'left'):
            #Player exits left
            self.x, self.y = ROOM_WIDTH - 2, ROOM_HEIGHT // 2
            room.entities.append(self)
            return room

        elif(transition_direction == 'bottom'):
            #Player exits bottom
            self.x, self.y = ROOM_WIDTH // 2, 1
            room.entities.append(self)
            return room

class Enemy(Entity):
    def __init__(self, x, y, sprite):
        super().__init__(
            x,
            y,
            sprite=sprite,
            blocks_movement=True,
            name="ENEMY"
        )
        self.state = "idle"
        self.aggro_range = 5

        self.hp = 20

        self.attack = 5
        self.attack_speed = 2000
        self.init_attack_delay = 1000    #delay before enemy can attack after switching to attack state for dodging purposes
        self.last_attack_time = 0

        self.move_speed = 1000
        self.last_move_time = 0

        self.is_hit = False
        self.last_hit_time = 0
        self.hit_cooldown = 100

    def attempt_move(self, dx, dy, room):
        '''Movement check for enemy movement and room transitions. 
        Checks if enemy is trying to move onto a blockable tile to transition rooms, or if the tile they are trying to move onto is blocked.'''
        target_x = self.x + dx
        target_y = self.y + dy

        if not room.is_blocked(target_x, target_y):
            room.update_entity_position(self, self.x + dx, self.y + dy)
            self.move(dx, dy)

    def update(self, player, room):
        distance_to_player = abs(self.x - player.x) + abs(self.y - player.y)
        current_time = pygame.time.get_ticks()

        # -----------------------------
        # Simple state machine for enemy behavior based on distance to player
        # -----------------------------
        if distance_to_player <= 1:
            self.state = "attack"
        elif distance_to_player <= self.aggro_range:
            self.state = "chase"
        else:
            self.state = "idle"


        # ------------------------------
        # State behavior implementations
        # ------------------------------
        if self.state == "attack":
            if current_time - self.last_attack_time >= self.attack_speed and current_time - self.last_move_time >= self.init_attack_delay:
                player.hp -= self.attack
                player.last_hit_time = current_time
                player.is_hit = True
                print(f"Enemy attacks! Player HP: {player.hp}")
                self.last_attack_time = current_time
        elif self.state == "chase":
            if current_time - self.last_move_time >= self.move_speed:
                dx = 0
                dy = 0
                if player.x < self.x:
                    dx = -1
                elif player.x > self.x:
                    dx = 1
                elif player.y < self.y:
                    dy = -1
                elif player.y > self.y:
                    dy = 1

                self.attempt_move(dx, dy, room)

                self.last_move_time = current_time

        # ------------------------------------------
        # Additional logic for hit cooldowns
        # ------------------------------------------
        if current_time - self.last_hit_time >= self.hit_cooldown:
            self.is_hit = False
            self.last_hit_time = current_time


class Chest(Entity):
    def __init__(self, x, y, sprite):
        super().__init__(
            x,
            y,
            sprite=sprite,
            blocks_movement=True,
            name="CHEST"
        )

        self.opened = False

    def interact(self, player, room):
        if not self.opened:
            self.opened = True
            print("Chest opened!")


class HealingFountain(Entity):
    def __init__(self, x, y, sprite):
        super().__init__(
            x,
            y,
            sprite=sprite,
            blocks_movement=True,
            name="HEALING"
        )

    def interact(self, player, room):
        player.hp = player.max_hp
        print("You feel refreshed!")