from PIL import Image
from carver import crop
from sys import argv

if __name__ == "__main__":
    [image_path, crop_rows, crop_cols] = argv[1:]
    crop_rows = int(crop_rows)
    crop_cols = int(crop_cols)
    original_image = Image.open(image_path)

    cropped_image, marked_original_image = crop(
        original_image.copy(), original_image, crop_rows, crop_cols)
    cropped_image.save('cropped.jpeg', 'jpeg')
    marked_original_image.save('marked.jpeg', 'jpeg')
