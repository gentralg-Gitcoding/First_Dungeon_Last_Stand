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

    @param width: resize to custom width defauts to TILE_SIZE
    @param height: resize to custom height defauts to TILE_SIZE
    @param namehint: namehint for image defaults to empty string, used for debugging purposes to identify which image is which when loading
    '''
    img = pygame.image.load(path, namehint).convert_alpha()
    return pygame.transform.scale(img, (width, height))

def get_img_hitbox_mask(img, rgb_color=(255, 255, 255), special_flags=pygame.BLEND_RGB_MAX, unsetcolor=(0, 0, 0, 0), setcolor=(255, 255, 255, 255)):
    '''
    Get a mask for an image to be used for pixel perfect collision detection.

    @param img: the image to get the hitbox mask for
    @param rgb_color: the color to fill the hitbox mask with, default is white
    @param special_flags: the special flags to use when filling the hitbox mask, default is pygame.BLEND_RGB_MAX which will make the hitbox white
    @param unsetcolor: the color to set as transparent in the hitbox mask, default is (0, 0, 0, 0) which is fully transparent
    @param setcolor: the color to set as opaque in the hitbox mask, default is (255, 255, 255, 255) which is fully opaque
    '''
    img_hit = img.copy()
    mask = pygame.mask.from_surface(img_hit)
    img_hit.fill(rgb_color, special_flags=special_flags)  # Make the hit version of the sprite white to indicate being hit, can be changed to red or something else later for better feedback.
    img_hit = mask.to_surface(img_hit, unsetcolor=unsetcolor, setcolor=setcolor)
    return img_hit
