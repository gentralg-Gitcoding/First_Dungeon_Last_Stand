import pygame
from settings import TILE_SIZE

def get_img_frame_surface(img, frame):
    surface = img.subsurface(frame).copy()
    surface = pygame.transform.scale(surface, (TILE_SIZE, TILE_SIZE))   #scale to tile size
    return surface