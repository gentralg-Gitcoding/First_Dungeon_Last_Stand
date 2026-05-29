import pygame

class SoundManager():
    def __init__(self):
        self.sounds = {}
        self.sfx_volume = 0.5

    def load_sound(self, name, path):
        sound = pygame.mixer.Sound(path)
        sound.set_volume(self.sfx_volume)

        self.sounds[name] = sound

    def play_sfx(self, name):
        if name in self.sounds:
            self.sounds[name].play()