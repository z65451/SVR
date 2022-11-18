from numpy import ones, stack, array, rot90
# from numba import jit
from PIL import Image
from seamcarving.api.algorithm import apply_sobel, compute_optimal_seam, DIAGONAL_LEFT, DIAGONAL_RIGHT


# @jit
def trace_seam(mask, original_image, energy_image, seam_start, next_seam_position):
    seam_pos = seam_start
    while True:
        row, col = seam_pos
        mask[row][col] = 0
        original_image[row][col] = (255, 0, 0)
        energy_image[row][col] = (255, 0, 0)
        if (next_seam_position[row][col] == 0):
            break
        if (next_seam_position[row][col] == DIAGONAL_LEFT):
            seam_pos = (row + 1, col - 1)
        elif (next_seam_position[row][col] == DIAGONAL_RIGHT):
            seam_pos = (row + 1, col + 1)
        else:
            seam_pos = (row + 1, col)


def get_energy_image(image_to_crop: Image) -> Image:
    grayscale_to_crop = image_to_crop.convert('1')
    grayscale_to_crop_bytes = array(grayscale_to_crop)
    grayscale_to_crop_energy = apply_sobel(array(grayscale_to_crop_bytes))
    return Image.fromarray(grayscale_to_crop_energy)


def get_energy_in_rgb(energy_grayscale: array) -> array:
    return array(Image.fromarray(energy_grayscale).convert('RGB'))


def carve_column_and_mark_seam(image_to_crop: array, original_image: array, energy_image: array, seam_start: tuple, next_seam_position: tuple):
    rows, cols, _ = image_to_crop.shape
    mask = ones((rows, cols), dtype=bool)
    trace_seam(
        mask, original_image, energy_image, seam_start, next_seam_position
    )
    mask = stack([mask] * 3, axis=2)
    carved_image = image_to_crop[mask].reshape((rows, cols - 1, 3))
    return carved_image, original_image, energy_image


def crop_column(image_to_crop: Image, original_image: Image, energy_image: Image): #-> (Image, Image, Image):
    image_to_crop_energy = array(
        get_energy_image(image_to_crop)
    )

    (seam_start, seam_cost, next_seam_position) = compute_optimal_seam(
        image_to_crop_energy
    )

    resized, original, energy = carve_column_and_mark_seam(
        array(image_to_crop),
        array(original_image),
        get_energy_in_rgb(array(energy_image)),
        seam_start,
        next_seam_position
    )

    current_image = Image.fromarray(resized, 'RGB')
    original_image = Image.fromarray(original, 'RGB')
    energy_image = Image.fromarray(energy).convert('RGB')

    return current_image, original_image, energy_image


def crop_row(image_to_crop: Image, original_image: Image, energy_image: Image):# -> (Image, Image, Image):
    transposed_image_to_crop = image_to_crop.rotate(90, expand=True)
    transposed_original_image = original_image.rotate(90, expand=True)
    transposed_energy_image = energy_image.rotate(90, expand=True)

    cropped_image, marked_original_image, energy_image = crop_column(
        transposed_image_to_crop, transposed_original_image, transposed_energy_image,
    )

    return (
        cropped_image.rotate(3*90, expand=True),
        marked_original_image.rotate(3*90, expand=True),
        energy_image.rotate(3*90, expand=True)
    )


def crop(image_to_crop: Image, original_image: Image, energy_image: Image, rows: int, cols: int):# -> (Image, Image):
    cropped_image = image_to_crop
    marked_original_image = original_image
    marked_energy_image = energy_image

    for row in range(rows):
        cropped_image, marked_original_image, marked_energy_image = crop_row(
            cropped_image, marked_original_image, marked_energy_image
        )
    for col in range(cols):
        cropped_image, marked_original_image, marked_energy_image = crop_column(
            cropped_image, marked_original_image, marked_energy_image
        )
    return (cropped_image, marked_original_image, marked_energy_image)
