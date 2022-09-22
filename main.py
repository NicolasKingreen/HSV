import matplotlib.pyplot as plt
import numpy as np
from scipy import misc, ndimage

from math import cos, sin


def to_polar(x, y):
    magnitude = (x**2+y**2)**0.5
    angle = np.arctan2(y, x) * 180 / np.pi
    return magnitude, angle


def to_cartesian(radius, angle):
    return int(radius * cos(angle)), int(radius * sin(angle))


def clip(value, min_val, max_val):
    if value < min_val:
        value = min_val
    elif value > max_val:
        value = max_val
    return value


def transform_old(arr):
    width, height, colors = arr.shape
    print(width, height, colors)

    polar_arr = np.zeros(arr.shape)

    for x, row in enumerate(arr):
        for y, color in enumerate(row):
            radius, angle = to_polar(x, y)

            r, g, b = color
            r_rad, g_ang = to_polar(r, g)
            # r_rad = clip(r_rad * angle, 0, 255)
            new_r = clip(r_rad, 0, 255)
            new_g = clip(g_ang, 0, 255)
            new_b = clip(b, 0, 255)

            color = (new_r, new_g, new_b)

            # cell = clip(cell * angle, 0, 255)

            # angle *= radius
            # radius *= angle
            # radius = 100

            # x, y = to_cartesian(radius, angle)
            # x = clip(x + width // 2, 0, width-1)
            # y = clip(y + height // 2, 0, height-1)

            polar_arr[x, y] = color

    xs = np.arange(width)
    ys = np.arange(height)

    return polar_arr


def transform(image):

    width, height = image.shape[:2]

    reds = image[:, :, 0]
    greens = image[:, :, 1]
    blues = image[:, :, 2]

    radius = np.power((np.power(image, 2) + np.power(image, 2)), 0.5)
    angle = np.atan2(image, image)

    angle += 3.54

    return image


image = misc.face()

# image_gray = np.array(image[:, :, 0])
# lx, ly = image_gray.shape  # image size


# image_gray[range(lx), range(lx)] = 255  # diagonal line

# image_gray = transform(image_gray)
image = transform(image)

plt.imshow(image)
plt.show()
