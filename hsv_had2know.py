from math import acos, sqrt, pi


def rgb_to_hsv(color):
    M = max(color)
    m = min(color)
    C = M - m  # chroma

    V = M / 255
    S = 1 - m / M if M > 0 else 0

    R, G, B = color
    if G >= B:
        H = acos((R - 0.5*G - 0.5*B)/sqrt(R**2 + G**2 + B**2 - R*G - R*B - G*B)) * 180 / pi
        # print("H value is", H)
    elif B > G:
        H = 360 - acos((R - 0.5 * G - 0.5 * B)/sqrt(R**2 + G**2 + B**2 - R*G - R*B - G*B)) * 180 / pi

    return H, S, V


def hsv_to_rgb(color):
    H, S, V = color

    M = 255 * V
    m = M * (1-S)

    z = (M-m)*(1 - abs((H/60) % 2 - 1))

    # print(H)
    # H *= 180 / pi
    # print(H)

    if 0 <= H < 60:
        R = M
        G = z + m
        B = m
    elif 60 <= H < 120:
        R = z + m
        G = M
        B = m
    elif 120 <= H < 180:
        R = m
        G = M
        B = z + m
    elif 180 <= H < 240:
        R = m
        G = z + m
        B = M
    elif 240 <= H < 300:
        R = z + m
        G = m
        B = M
    elif 300 <= H < 360:
        R = M
        G = m
        B = z + m
    return int(R), int(G), int(B)


test_colors = [(255, 240, 232), (1, 1, 1), (127, 63, 1), (255, 1, 255)]
# test_colors_hsv = [rgb_to_hsv(color) for color in test_colors]

print(test_colors)
# print(test_colors_hsv)
# print([hsv_to_rgb(color) for color in test_colors_hsv])
print(hsv_to_rgb(rgb_to_hsv(test_colors[0])))

tomato_color = (255, 101, 66)
tomato_color_hsv = rgb_to_hsv(tomato_color)
print(hsv_to_rgb(tomato_color_hsv))



