import pygame

class MusicManager():
    def __init__(self):
        self.music_volume = 0.2
        self.music_library = {}


    def play_music(self, path, loops=-1):
        pygame.mixer.music.load(path)
        pygame.mixer.music.set_volume(self.music_volume)
        pygame.mixer.music.play(loops)