from carver import crop, get_energy_image
from flask import Flask, jsonify, request
from flask_cors import CORS

from numpy import array
from PIL import Image
from uuid import uuid4
from os import path
from io import BytesIO
from io import StringIO
import base64
import re
app = Flask(__name__, static_folder='resized-images')
cors = CORS(app)
app_root = path.dirname(path.abspath(__file__))

bad_request_status_code = 201


@app.route('/')
def home():
    return 'api running...'


@app.route('/resize', methods=['POST'])
def resize():
    errors = validate_resize_request(request)
    if len(errors):
        return jsonify(errors), bad_request_status_code

    image_data = re.sub('^data:image/.+;base64,', '', request.form['image'])
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')

    [crop_rows, crop_cols] = [
        int(request.form['crop_rows']),
        int(request.form['crop_cols'])
    ]

    processed_images = process_resize_request(image, crop_rows, crop_cols)
    return jsonify({
        "processed_images": processed_images,
    }), 200


def validate_resize_request(request):
    errors = []
    if 'image' not in request.form:
        errors.append('Image must be present.')
    if 'crop_rows' not in request.form or not is_number(request.form['crop_rows']):
        errors.append(
            'crop_rows must be present in request and it must be an integer')
    if 'crop_cols' not in request.form or not is_number(request.form['crop_cols']):
        errors.append(
            'crop_cols must be present in request and it must be an integer')
    return errors


def process_resize_request(image: Image, crop_rows: int, crop_cols: int):
    original_energy_image = get_energy_image(image)
    cropped_image, marked_original_image, marked_energy_image = crop(
        image.copy(),
        image,
        original_energy_image.copy(),
        crop_rows, crop_cols
    )
    generated_names = get_images_names()
    [
        original_name,
        cropped_name,
        marked_name,
        energy_name,
        marked_energy_name
    ] = get_images_paths(generated_names)

    marked_original_image.save(marked_name)
    image.save(original_name)
    cropped_image.save(cropped_name)
    original_energy_image.convert('RGB').save(energy_name)
    marked_energy_image.convert('RGB').save(marked_energy_name)

    return get_images_urls(generated_names)


def get_images_paths(names: list) -> list:
    return [path.join(app_root, 'resized-images', name) for name in names]


def get_images_urls(names: list) -> list:
    return ['http://{0}/resized-images/{1}'.format(request.host, name) for name in names]


def get_images_names():
    image_name = str(uuid4())
    return [
        'original-{0}.jpeg'.format(image_name),
        'cropped-{0}.jpeg'.format(image_name),
        'marked-{0}.jpeg'.format(image_name),
        'energy-{0}.jpeg'.format(image_name),
        'energy-marked-{0}.jpeg'.format(image_name)
    ]


def is_number(string: str) -> bool:
    return string.replace('.', '', 1).isdigit()
