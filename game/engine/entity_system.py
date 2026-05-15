import pygame

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

    def update(self, world):
        """
        Called every game tick/frame.
        Override in subclasses.
        """
        pass

    def interact(self, player, world):
        """
        Called when the player interacts with entity.
        Override in subclasses.
        """
        pass

    def render(self, screen, tile_size):
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

        self.hp = 20

    def update(self, world):
        # Enemy AI later
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

    def interact(self, player, world):
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

    def interact(self, player, world):
        player.hp = 100
        print("Player healed!")