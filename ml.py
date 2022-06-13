from operator import le
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def line_vector(line, scale=1):
    return (int((line[1][0] - line[0][0]) * scale), int((line[1][1] - line[0][1]) * scale))


def line_moved(line, offset):
    return ((line[0][0] + offset[0], line[0][1] + offset[1]), (line[1][0] + offset[0], line[1][1] + offset[1]))


def line_scaled(line, scale):
    v = line_vector(line)
    return (line[0], (line[0][0] + int(v[0] * scale), line[0][1] + int(v[1] * scale)))

def render(image, tile, van, floor_mask):
    height, width = image.shape[0], image.shape[1]

    buffer = np.copy(image)
    scale = 3
    left = (van, (int(width * -(scale - 1)), height))
    right = (van, (int(width * (scale)), height))

    # Алгоритм A
    # med = (van, line_intersection(((0,height),(width,height)),line_rotated(left, (line_angle(right) - line_angle(left)) / 2)))
    # top = line_moved(line_rotated(med), line_vector(med, 0.5))
    # bottom = line_moved(line_rotated(med), line_vector(med, 0.9))
    # tl = line_intersection(left, top)
    # tr = line_intersection(top, right)
    # br = line_intersection(right, bottom)
    # bl = line_intersection(bottom, left)

    # Алгоритм Б
    tl = (int(line_scaled(left, 0.2)[1][0]), int(line_scaled(left, 0.2)[1][1]))
    bl = left[1]
    tr = (int(line_scaled(right, 0.2)[1][0]), int(line_scaled(right, 0.2)[1][1]))
    br = right[1]

    tile_width = tile.shape[1]
    tile_height = tile.shape[0]
    src = np.float32([(0, 0), (tile_width, 0), (tile_width, tile_height), (0, tile_height)])

    dst = np.float32([tl, tr, br, bl])
    M = cv2.getPerspectiveTransform(src, dst)
    floor = cv2.warpPerspective(tile, M, (width, height), flags=cv2.INTER_LINEAR)
    _, mask = cv2.threshold(floor[:, :, 0], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Убираем дырки
    kernel = np.ones((5, 5), 'uint8')
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    # Совмещаем слои
    mask = cv2.bitwise_and(mask, floor_mask)
    buffer[np.where(mask > 0)] = floor[np.where(mask > 0)]
    return buffer


