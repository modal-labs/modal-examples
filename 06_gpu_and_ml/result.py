import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Game variables
gravity = 0.5
bird_y_velocity = 0
bird_jump = -5
bird_size = 20
ground_height = 20
pipe_width = 30
pipe_gap = 100
pipe_color_options = [(60, 80,