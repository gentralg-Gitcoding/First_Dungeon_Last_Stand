import pygame

class GameStateSystem:
    def __init__(self):
        self.state = "playing"  # i.e. "playing", "game_over", "victory"

    def update(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.state == "playing":
                print("Game paused.")
                self.state = "paused"
            elif self.state == "paused":
                print("Game resumed.")
                self.state = "playing"
        if event.type == pygame.QUIT:
            print("Quitting game.")
            self.state = "quit"
