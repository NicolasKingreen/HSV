import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

from math import sqrt, sin, cos, pi

import pygame
from pygame.locals import *


def to_spherical(x, y, z):
    r = sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)
    return r, theta, phi


def to_cartesian(r, theta, phi):
    x = r * cos(phi) * sin(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(theta)
    return x, y, z


def to_spherical_np(np_array):
    reds = np_array[:, :, 0]
    greens = np_array[:, :, 1]
    blues = np_array[:, :, 2]

    rs = np.sqrt(np.power(reds, 2) +
                 np.power(greens, 2) +
                 np.power(blues, 2))
    thetas = np.arctan2(np.sqrt(np.power(reds, 2) +
                                np.power(greens, 2)),
                        blues)
    phis = np.arctan2(greens, reds)

    combined = np.dstack((rs, thetas, phis))
    return combined


def to_cartesian_np(np_array):
    rs = np_array[:, :, 0]
    thetas = np_array[:, :, 1]
    phis = np_array[:, :, 2]

    xs = rs * np.cos(phis) * np.sin(thetas)
    ys = rs * np.sin(phis) * np.sin(thetas)
    zs = rs * np.cos(thetas)

    combined = np.dstack((xs, ys, zs))
    return combined


def transform_image(image, delta_time):
    spherical_img = to_spherical_np(image.astype(int))

    spherical_img[:, :, 1] *= (1 * delta_time)
    spherical_img[:, :, 2] *= (2 * delta_time)

    cartesian_np = to_cartesian_np(spherical_img)
    np.clip(cartesian_np, 0, 255)
    return cartesian_np


def main_loop(window_size):
    pygame.init()
    screen = pygame.display.set_mode(window_size)
    clock = pygame.time.Clock()
    pg_image = np.rot90(image, k=1)
    surface = pygame.surfarray.make_surface(pg_image)

    is_running = True
    while is_running:

        frame_time_s = clock.tick()
        fps = clock.get_fps()
        print(fps)

        for event in pygame.event.get():
            if event.type == QUIT:
                is_running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    is_running = False

        screen.fill("white")
        surface = pygame.surfarray.make_surface(transform_image(pg_image, frame_time_s))
        screen.blit(surface, (0, 0))
        pygame.display.update()


image = misc.face()
image_size = image.shape[1], image.shape[0]
print(image_size)
print(image.shape)

# image = image[:3, :3]
# spherical_image = np.zeros(image.shape)
# for x, row in enumerate(spherical_image):
#     for y, cell in enumerate(row):
#         spherical_image[x, y] = to_spherical(*image[x, y])

# print(image == to_cartesian_np(to_spherical_np(image)))

print("Original")
print(image[0, 0])
print()

print("One pixel spherical")
spherical_pixel = to_spherical(*image[0, 0])
print(list(spherical_pixel))

print("One pixel back to cartesian")
cartesian_pixel = to_cartesian(*spherical_pixel)
print([int(d) for d in cartesian_pixel])
print()

print("NumPy spherical")
spherical_np = to_spherical_np(image)
print(spherical_np[0, 0])

print("Transformation")
spherical_np[:, :, 1] *= 5

print("NumPy back to cartesian")
cartesian_np = to_cartesian_np(spherical_np).astype(int)
np.clip(cartesian_np, 0, 255)
print(cartesian_np[0, 0])

main_loop(image_size)
# plt.imshow(cartesian_np)
# plt.show()
