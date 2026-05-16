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
            name="player"
        )

        self.hp = 100
        self.attack = 10
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
            name="enemy"
        )
        self.state = "idle"

        self.hp = 20
        self.attack = 5

        self.move_delay = 300
        self.last_move_time = 0

        self.aggro_range = 5

    def update(self, player, room):
        pass


class Chest(Entity):
    def __init__(self, x, y, sprite):
        super().__init__(
            x,
            y,
            sprite=sprite,
            blocks_movement=True,
            name="loot"
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
            blocks_movement=False,
            name="healing"
        )

    def interact(self, player, room):
        player.hp = 100
        print("You feel refreshed!")