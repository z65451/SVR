from numpy import ones, stack, array, rot90
from numba import jit
from PIL import Image
from algorithm import apply_sobel, compute_optimal_seam


@jit
def trace_seam_into_mask(mask, original_image, seam_start, next_seam_position):
    seam_pos = seam_start
    while True:
        row, col = seam_pos
        mask[row][col] = 0
        original_image[row][col] = (255, 0, 0)
        if (next_seam_position[row][col] == 0):
            break
        if (next_seam_position[row][col] == 1):
            seam_pos = (row + 1, col + 1)
        else:
            seam_pos = (row + 1, col)
    return mask


def carve_column_and_mark_seam(image, original_image, seam_start, next_seam_position):
    rows, cols, _ = image.shape
    mask = ones((rows, cols), dtype=bool)
    mask = trace_seam_into_mask(
        mask, original_image, seam_start, next_seam_position
    )
    mask = stack([mask] * 3, axis=2)
    carved_image = image[mask].reshape((rows, cols - 1, 3))
    return carved_image, original_image


def crop_column(image_to_crop: Image, original_image: Image) -> Image:
    grayscale_image = image_to_crop.convert('1')
    image_bytes = array(grayscale_image)
    energy = apply_sobel(image_bytes)
    (seam_start, seam_cost, next_seam_position) = compute_optimal_seam(energy)
    resized, original = carve_column_and_mark_seam(
        array(image_to_crop),
        array(original_image),
        seam_start,
        next_seam_position
    )
    current_image = Image.fromarray(resized, 'RGB')
    original_image = Image.fromarray(original, 'RGB')
    return current_image, original_image


def crop_row(image_to_crop: Image, original_image: Image) -> (Image, Image):
    transposed_image_to_crop = image_to_crop.rotate(90, expand=True)
    transposed_original_image = original_image.rotate(90, expand=True)
    cropped_image, marked_original_image = crop_column(
        transposed_image_to_crop, transposed_original_image
    )
    return (
        cropped_image.rotate(3*90, expand=True),
        marked_original_image.rotate(3*90, expand=True)
    )


def crop(image_to_crop: Image, original_image: Image, rows: int, cols: int) -> (Image, Image):
    marked_original_image = original_image
    for row in range(rows):
        image_to_crop, marked_original_image = crop_row(
            image_to_crop, marked_original_image
        )
    for col in range(cols):
        image_to_crop, marked_original_image = crop_column(
            image_to_crop, marked_original_image
        )
    return image_to_crop, marked_original_image
