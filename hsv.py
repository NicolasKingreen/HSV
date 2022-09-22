import numpy as np
import matplotlib.pyplot as plt

from math import ceil
from PIL import Image


def rgb_to_hsv(color):
    R, G, B = color

    M = max(color)
    m = min(color)
    C = M - m

    if C == 0:
        H0 = 0
    elif M == R:
        H0 = ((G-B)/C) % 6
    elif M == G:
        H0 = (B-R)/C + 2
    elif M == B:
        H0 = (R-G)/C + 4
    H = 60 * H0  # in degrees

    # alpha = 0.5 * (2*R - G - B)
    # beta = sqrt(3)/2 * (G - B)
    # H2 = atan2(beta, alpha)
    # C2 = sqrt(alpha**2 + beta**2)

    V = M
    # L = 0.5 * (M + m)
    Sv = 0 if V == 0 else C / V
    # Sl = 0 if L == 1 or L == 0 else C/(1-abs(2*L-1))

    return int(H), Sv, V / 255

def hsv_to_rgb(color):
    H, S, V = color
    C = V * S
    H0 = H / 60
    X = C * (1 - abs(H0 % 2 - 1))
    if 0 <= H0 < 1:
        R, G, B = C, X, 0  # 1
    elif 1 <= H0 < 2:
        R, G, B = X, C, 0
    elif 2 <= H0 < 3:
        R, G, B = 0, C, X  # 1
    elif 3 <= H0 < 4:
        R, G, B = 0, X, C
    elif 4 <= H0 < 5:
        R, G, B = X, 0, C  # 1
    elif 5 <= H0 < 6:
        R, G, B = C, 0, X

    # seg_i = trunc(H0)
    # R, G, B = np.roll(np.array([X, C, 0]), trunc(H0)-1)//2)
    # if seg_i % 2 else R, G, B = deque([C, X, 0]).rotate(trunc(H0)//2)

    m = V - C
    R = (R + m) * 255
    G = (G + m) * 255
    B = (B + m) * 255
    return ceil(R), ceil(G), ceil(B)


def rgb_to_hsv_vectorised(a):
    a = a.astype("float")
    Rs, Gs, Bs = a[:, :, 0], a[:, :, 1], a[:, :, 2]

    Ms = a.max(axis=2)
    Mis = a.argmax(axis=2)
    ms = a.min(axis=2)

    Cs = Ms - ms

    Hs = np.full(Rs.shape, np.inf)
    Hs = np.where(Mis == 0, ((Gs-Bs)/Cs) % 6, Hs)  # max is red
    Hs = np.where(Mis == 1, ((Bs-Rs)/Cs) + 2, Hs)  # max is green
    Hs = np.where(Mis == 2, ((Rs-Gs)/Cs) + 4, Hs)  # max is blue
    Hs = np.where(Hs == np.inf, 0, Hs)
    Hs *= 60

    Vs = Ms
    Ss = np.where(Vs == 0, 0, Cs/Vs)

    HSV = np.dstack((Hs, Ss, Vs/255))
    return HSV


def hsv_to_rgb_vectorized(a):
    Hs, Ss, Vs = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    Cs = Vs * Ss
    Hs /= 60
    Xs = Cs * (1-np.abs(np.mod(Hs, 2) - 1))

    Rs = np.zeros(Hs.shape)
    Rs = np.where(((0 <= Hs) & (Hs < 1)) | ((5 <= Hs) & (Hs < 6)), Cs, Rs)
    Rs = np.where(((1 <= Hs) & (Hs < 2)) | ((4 <= Hs) & (Hs < 5)), Xs, Rs)

    Gs = np.zeros(Hs.shape)
    Gs = np.where((1 <= Hs) & (Hs < 3), Cs, Gs)
    Gs = np.where(((0 <= Hs) & (Hs < 1)) | ((3 <= Hs) & (Hs < 4)), Xs, Gs)

    Bs = np.zeros(Hs.shape)
    Bs = np.where((3 <= Hs) & (Hs < 5), Cs, Bs)
    Bs = np.where(((2 <= Hs) & (Hs < 3)) | ((5 <= Hs) & (Hs < 6)), Xs, Bs)

    ms = Vs - Cs
    Rs = np.ceil((Rs + ms) * 255)
    Gs = np.ceil((Gs + ms) * 255)
    Bs = np.ceil((Bs + ms) * 255)

    RGB = np.dstack((Rs, Gs, Bs)).astype("uint8")
    return RGB


def hsv_transform(a):

    # hue rotation
    a[..., 0] += 60
    a[..., 0] = np.where(a[..., 0] > 360, a[..., 0] % 360, a[..., 0])
    # a[..., 0] %= 360

    # saturation
    a[..., 1] += 0.3
    a[..., 1] = np.where(a[..., 1] > 1, 1, a[..., 1])

    # value
    a[..., 2] -= 0.3
    a[..., 2] = np.where(a[..., 2] < 0, 0, a[..., 2])

    # swapping values
    a[..., 1], a[..., 2] = a[..., 2], a[..., 1]

    return a


def hsv_gradient(a):
    height, width = a.shape[:2]
    ratio = width / height

    # horizontal gradient
    horizontal_gradient_vector = np.linspace(0, 1, width)
    horizontal_gradient_matrix = np.tile(horizontal_gradient_vector, (height, 1))
    # a[..., 1] = horizontal_gradient_matrix

    # vertical gradient
    vertical_gradient_vector = np.linspace(0, 1, height).reshape((height, 1))
    vertical_gradient_matrix = np.tile(vertical_gradient_vector, (1, width))
    # a[..., 2] = vertical_gradient_matrix

    # diagonal gradient (via summing vertical and horizontal vectors)
    a[..., 1] = (horizontal_gradient_matrix + vertical_gradient_matrix) / 2
    # a[..., 2] = (horizontal_gradient_matrix + vertical_gradient_matrix) / 2

    return a


# test_colors = [(0, 0, 0), (255, 255, 255), (255, 0, 255), (255, 101, 66), (101, 99, 56)]
# test_colors_hsv = [rgb_to_hsv(color) for color in test_colors]
#
# print(test_colors_hsv)
# print([hsv_to_rgb(color) for color in test_colors_hsv])

# print(dir(rgb_to_hsv))
# print(rgb_to_hsv.__code__)

img = Image.open("image.jpg")
pixels = np.array(img)
# plt.imshow(pixels)
# plt.show()

hsv_img = rgb_to_hsv_vectorised(pixels)
# hsv_img = hsv_transform(hsv_img)
hsv_img = hsv_gradient(hsv_img)
back_and_forth = hsv_to_rgb_vectorized(hsv_img)
plt.imshow(back_and_forth)
plt.show()
