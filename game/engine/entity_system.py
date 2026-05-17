import pygame

from settings import TILE_SIZE

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

    def render(self, screen, tile_size=TILE_SIZE):
        """
        Draw entity to screen.
        """
        if self.sprite:
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
            blocks_movement=True,
            name="PLAYER"
        )

        self.hp = 100
        self.attack = 10
        self.move_delay = 100
        self.last_move_time = 0
        self.transition_cooldown = 0
        self.facing = "down"


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

        self.hp = 20
        self.attack = 5
        self.attack_speed = 2000

        self.move_delay = 1000
        self.last_move_time = 0

        self.aggro_range = 5

    def attempt_move(self, dx, dy, room):
        '''Movement check for enemy movement and room transitions. 
        Checks if enemy is trying to move onto a blockable tile to transition rooms, or if the tile they are trying to move onto is blocked.'''
        target_x = self.x + dx
        target_y = self.y + dy

        #Check which way the player went
        # transition_direction = get_direction(dx, dy)

        #Check if target is a door and if so, handle room transition
        # if room.room_map[target_y][target_x] == ROOM_TILE_DICT['DOOR']:
        #     (entity.x, entity.y), room_pos, room = handle_room_transition(entity.position, transition_direction, room_pos, room)
        #     return entity.x, entity.y, room_pos, room
        if not room.is_blocked(target_x, target_y):
            room.update_entity_position(self, self.x + dx, self.y + dy)
            self.move(dx, dy)
            # return True
        # else:
        #     return False

    def update(self, player, room):
        distance_to_player = abs(self.x - player.x) + abs(self.y - player.y)
        current_time = pygame.time.get_ticks()

        if distance_to_player <= 1:
            self.state = "attack"
        elif distance_to_player <= self.aggro_range:
            self.state = "chase"
        else:
            self.state = "idle"

        if self.state == "attack":
            if current_time - self.last_move_time >= self.attack_speed:
                player.hp -= self.attack
                print(f"Enemy attacks! Player HP: {player.hp}")
                self.last_move_time = current_time
        elif self.state == "chase":
            if current_time - self.last_move_time >= self.move_delay:
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
        player.hp = 100
        print("You feel refreshed!")