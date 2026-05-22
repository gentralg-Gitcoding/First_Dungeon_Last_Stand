import pygame
from settings import TILE_SIZE

class SpriteSheet:
    def __init__(self, img, matrix_size, frame_width, frame_height):
        self.img = img
        self.matrix_size = matrix_size
        self.frame_width = frame_width
        self.frame_height = frame_height

    def get_sprite_sheet_frame(self, row, col, offset=(0, 0), width_multiplier=1, height_multiplier=1):
        """
        Extract a frame from a sprite sheet.

        @param img: the sprite sheet image
        @param matrix_size: (frame_width, frame_height) in pixels of each frame in the sprite sheet
        @param frame_width: width of the frame to extract in pixels
        @param frame_height: height of the frame to extract in pixels
        @param row: animation row
        @param col: animation frame
        @param offset: Optional (x, y) offset in pixels to apply to the frame position
        @param width_multiplier: Optional how many tiles wide the final surface should be
        @param height_multiplier: Optional how many tiles tall the final surface should be

        @return: a pygame Surface of the extracted frame, scaled to tile size

        """

        rect = pygame.Rect(
            col * self.matrix_size[0],
            row * self.matrix_size[1], 
            self.frame_width, 
            self.frame_height
        ).move(offset)

        surface = self.img.subsurface(rect)
        surface = pygame.transform.scale(surface, (TILE_SIZE * width_multiplier, TILE_SIZE * height_multiplier))   #scale to tile size

        return surface