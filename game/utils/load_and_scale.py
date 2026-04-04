import pygame
from settings import TILE_SIZE

def load_img(path, namehint=''):
    '''
    Load an image keeping the original size.
    '''
    return pygame.image.load(path, namehint).convert_alpha()

def load_and_scale(path, width=TILE_SIZE, height=TILE_SIZE, namehint=''):
    '''
    Load and scale an image to the standard 32 by 32 tile size for the game.
    Optional: 
        width: resize to custom width
        height: resize to custom height
        namehint: namehint for image
    '''
    img = pygame.image.load(path, namehint).convert_alpha()
    return pygame.transform.scale(img, (width, height))

def resize_surface(surface, width=TILE_SIZE, height=TILE_SIZE):
    return pygame.transform.scale(surface, (width, height))