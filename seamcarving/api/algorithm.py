from numpy import array, sqrt, max, zeros_like, int, argmin
from scipy.signal import convolve2d, gaussian

from collections import deque
from sys import maxsize
from typing import List
import numpy

sobel_kernels = {
    'x': array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]),
    'y': array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
}

gaussian_kernel = (1/16) * array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])

DIAGONAL_LEFT = numpy.intp(1)
DOWN = numpy.intp(2)
DIAGONAL_RIGHT = numpy.intp(3)


def apply_sobel(image: array):
    blurred = convolve2d(image, gaussian_kernel, mode='same', boundary='symm')
    grad_x = convolve2d(
        blurred, sobel_kernels['x'], mode='same', boundary='symm')
    grad_y = convolve2d(
        blurred, sobel_kernels['y'], mode='same', boundary='symm')
    grad = sqrt(grad_x * grad_x + grad_y * grad_y)
    normalised_grad = grad * (255.0 / max(grad))
    return normalised_grad


def is_in_image(position: tuple, rows: int, cols: int) -> bool:
    row, col = position
    return row >= 0 and col >= 0 and row < rows and col < cols


def compute_optimal_seam(energy):
    rows, cols = energy.shape
    infinity = maxsize / 10
    dp = energy.copy()

    next_seam_position = zeros_like(dp, dtype=numpy.intp)

    for col in range(cols):
        dp[rows - 1][col] = energy[rows-1][col]

    for row in range(rows - 2, -1, -1):
        for col in range(cols):
            optimal_adjacent_cost = infinity
            optimal_choice = -1
            adjacents = [
                ((row + 1, col - 1), DIAGONAL_LEFT),
                ((row + 1, col), DOWN),
                ((row + 1, col + 1), DIAGONAL_RIGHT),
            ]
            for (adjacent, choice) in adjacents:
                adjacent_row, adjacent_col = adjacent
                if not is_in_image(adjacent, rows, cols):
                    continue
                if dp[adjacent_row][adjacent_col] < optimal_adjacent_cost:
                    optimal_adjacent_cost = dp[adjacent_row][adjacent_col]
                    optimal_choice = choice

            next_seam_position[row][col] = optimal_choice
            dp[row][col] = energy[row][col] + optimal_adjacent_cost

    seam_start_col = argmin(dp[0, :])
    seam_start = (0, seam_start_col)
    seam_cost = dp[0][seam_start_col]
    return (seam_start, seam_cost, next_seam_position)
