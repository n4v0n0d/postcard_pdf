import numpy as np

from io import BytesIO
from itertools import batched
from pathlib import Path
from PIL import Image
from pypdf import PdfReader, PdfWriter, Transformation

SOURCE_FOLDER = Path('sources')
RESULTS_FOLDER = Path('results')

PAD_COLOR = (0, 0, 0, 0)
CUT_LINES_COLOR = (255, 0, 0, 255)

DPI = 300

def inch_to_pixel(inches):
    # Also make it an even number
    return int((inches * DPI) / 2) * 2

BLEED_SIZE = inch_to_pixel(0.10)
CUT_LINE_LENGTH = inch_to_pixel(1)

# offset cut lines 2 px away from bleed
CUT_LINE_OFFSET = 2

TARGET_WIDTH = inch_to_pixel(8.5)
TARGET_HEIGHT = inch_to_pixel(11)
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)


def get_image(name) -> Image:
    return Image.open(SOURCE_FOLDER / name)


def get_images():
    for path in SOURCE_FOLDER.iterdir:
        yield Image.open(path)


def save_to_results(image: Image, name):
    image.save(RESULTS_FOLDER / name)


def smear(arr: np.array, size, axis_index):
    # When we are taking only a single row of pixels from an edge of the image,
    # we lose a dimension, so we end up with an array shaped like e.g. [1500, 4]
    # where 1500 was the width and 4 is RGBA.
    # To smear it we have to add the missing dimension back.
    # Use axis index 0 for top and bottom smears, 1 for left and right.
    return np.repeat(np.expand_dims(arr, axis=axis_index), size, axis_index)


def single_pixel_smear(arr: np.array, size):
    # Smear a single RGBA pixel (should have shape [4]) to a square image.
    return smear(smear(arr, size, 0), size, 0)


def add_corner_bleed(image, bled_image, image_array, bleed_size) -> Image:
    half_bleed = int(bleed_size / 2)
    width, height = image.size

    top_left = single_pixel_smear(image_array[0, 0, :], half_bleed)
    top_right = single_pixel_smear(image_array[0, -1, :], half_bleed)
    bottom_left = single_pixel_smear(image_array[-1, 0, :], half_bleed)
    bottom_right = single_pixel_smear(image_array[-1, -1, :], half_bleed)

    bled_image.paste(Image.fromarray(top_left), (bleed_size - half_bleed, bleed_size - half_bleed))
    bled_image.paste(Image.fromarray(top_right), (width + bleed_size, bleed_size - half_bleed))
    bled_image.paste(Image.fromarray(bottom_left), (bleed_size - half_bleed, height + bleed_size))
    bled_image.paste(Image.fromarray(bottom_right), (width + bleed_size, height + bleed_size))

    return bled_image


def add_bleed_smear(image: Image, bleed_size) -> Image:
    width, height = image.size
    new_width = width + (2 * bleed_size)
    new_height = height + (2 * bleed_size)

    image_array = np.asarray(image)
    bled_image = Image.new('RGBA', (new_width, new_height), PAD_COLOR)
    bled_image.paste(image, (bleed_size, bleed_size))

    top = smear(image_array[0, :, :], bleed_size, 0)
    bottom = smear(image_array[-1, :, :], bleed_size, 0)
    left = smear(image_array[:, 0, :], bleed_size, 1)
    right = smear(image_array[:, -1, :], bleed_size, 1)

    bled_image.paste(Image.fromarray(top), (bleed_size, 0))
    bled_image.paste(Image.fromarray(bottom), (bleed_size, height + bleed_size))
    bled_image.paste(Image.fromarray(left), (0, bleed_size))
    bled_image.paste(Image.fromarray(right), (width + bleed_size, bleed_size))

    add_corner_bleed(image, bled_image, image_array, bleed_size)

    return bled_image


def add_bleed_flip(image: Image, bleed_size) -> Image:
    width, height = image.size
    new_width = width + (2 * bleed_size)
    new_height = height + (2 * bleed_size)

    image_array = np.asarray(image)
    bled_image = Image.new('RGBA', (new_width, new_height), PAD_COLOR)
    bled_image.paste(image, (bleed_size, bleed_size))

    top = flip_top_bottom(image.crop((0, 0, width, bleed_size)))
    bottom = flip_top_bottom(image.crop((0, height - bleed_size, width, height)))
    left = flip_left_right(image.crop((0, 0, bleed_size, height)))
    right = flip_left_right(image.crop((width - bleed_size, 0, width, height)))

    bled_image.paste(top, (bleed_size, 0))
    bled_image.paste(bottom, (bleed_size, height + bleed_size))
    bled_image.paste(left, (0, bleed_size))
    bled_image.paste(right, (width + bleed_size, bleed_size))

    add_corner_bleed(image, bled_image, image_array, bleed_size)

    return bled_image


def add_cut_lines(image: Image, bleed_size, line_length) -> Image:
    # assume image already had bleed added with same bleed_size

    # define relative coordinates, (0, 0) is the top left of the original image.
    def absolute(x, y):
        return x + line_length + bleed_size, y + line_length + bleed_size

    orig_width = image.size[0] - (2 * bleed_size)
    orig_height = image.size[1] - (2 * bleed_size)

    vertical_cut_line = Image.new('RGBA', (2, line_length), CUT_LINES_COLOR)
    horizontal_cut_line = Image.new('RGBA', (line_length, 2), CUT_LINES_COLOR)

    result_size = (image.size[0] + (2 * line_length), image.size[1] + (2 * line_length))
    result = Image.new('RGBA', result_size, PAD_COLOR)
    result.paste(image, (line_length, line_length))
    # top
    result.paste(vertical_cut_line,   absolute(CUT_LINE_OFFSET - 1,  -1*(bleed_size + line_length)))
    result.paste(vertical_cut_line,   absolute(orig_width - 1 - CUT_LINE_OFFSET,  -1*(bleed_size + line_length)))
    # bottom
    result.paste(vertical_cut_line,   absolute(CUT_LINE_OFFSET - 1,  orig_height + bleed_size))
    result.paste(vertical_cut_line,   absolute(orig_width - 1 - CUT_LINE_OFFSET,  orig_height + bleed_size))
    # left
    result.paste(horizontal_cut_line, absolute(-1*(bleed_size + line_length), CUT_LINE_OFFSET - 1))
    result.paste(horizontal_cut_line, absolute(-1*(bleed_size + line_length), orig_height - 1 - CUT_LINE_OFFSET))
    # right
    result.paste(horizontal_cut_line, absolute(orig_width + bleed_size, CUT_LINE_OFFSET - 1))
    result.paste(horizontal_cut_line, absolute(orig_width + bleed_size, orig_height - 1 - CUT_LINE_OFFSET))
    return result


def rotate_left(image: Image) -> Image:
    return image.transpose(Image.ROTATE_90)


def rotate_right(image: Image) -> Image:
    return image.transpose(Image.ROTATE_270)


def flip_left_right(image: Image) -> Image:
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def flip_top_bottom(image: Image) -> Image:
    return image.transpose(Image.FLIP_TOP_BOTTOM)


def bleed_and_stack(image_1, image_2, bleed_size, rotator) -> Image:
    def bleed_and_rotate(image) -> Image:
        return rotator(add_bleed(image, bleed_size))

    image_1 = bleed_and_rotate(image_1)
    image_2 = bleed_and_rotate(image_2)

    new_width = max(image_1.size[0], image_2.size[0])
    new_height = image_1.size[1] + image_2.size[1]
    stacked = Image.new('RGBA', (new_width, new_height), PAD_COLOR)
    stacked.paste(image_1, (0, 0))
    stacked.paste(image_2, (0, image_1.size[1]))

    cut = add_cut_lines(stacked, bleed_size, CUT_LINE_LENGTH)
    return cut


def bleed_stack_cut(image_1, image_2, bleed_size, rotator) -> Image:
    def transform_image(image) -> Image:
        return add_cut_lines(
            rotator(add_bleed(image, bleed_size)),
            bleed_size,
            CUT_LINE_LENGTH
        )

    image_1 = transform_image(image_1)
    image_2 = transform_image(image_2)
    # crop out the top cut lines on image 2.
    image_2 = image_2.crop((0, CUT_LINE_LENGTH, *image_2.size))

    new_width = max(image_1.size[0], image_2.size[0])
    new_height = image_1.size[1] + image_2.size[1] - (1 * CUT_LINE_LENGTH)
    result = Image.new('RGBA', (new_width, new_height), PAD_COLOR)
    result.paste(image_1, (0, 0))
    result.paste(
        image_2,
        (0, image_1.size[1] - (1 * CUT_LINE_LENGTH))
    )

    return result


def resize_centered(image: Image, target_size) -> Image:
    result = Image.new('RGBA', target_size, PAD_COLOR)
    paste_x = int((image.size[0] - target_size[0]) / 2)
    paste_y = int((image.size[1] - target_size[1]) / 2)
    breakpoint()
    result.paste(image, (paste_x, paste_y))
    return result


def process_images():
    # Assume an even number of images for now.
    for i, (path_1, path_2) in enumerate(batched(sorted(SOURCE_FOLDER.iterdir()), 2)):
        print(f'Processing batch {i}')
        image_1 = Image.open(path_1)
        image_2 = Image.open(path_2)

        for tater_name, rotater in taters:
            bsc_image = bleed_stack_cut(image_1, image_2, BLEED_SIZE, rotater)
            # bsc_image = resize_centered(bsc_image, TARGET_SIZE)
            crop_topleft_coord = (int((bsc_image.size[0] - TARGET_WIDTH) / 2), int((bsc_image.size[1] - TARGET_HEIGHT) / 2))
            crop_bottomright_coord = crop_topleft_coord[0] + TARGET_WIDTH, crop_topleft_coord[1] + TARGET_HEIGHT
            cropped = bsc_image.crop((*crop_topleft_coord, *crop_bottomright_coord))
            save_to_results(cropped, f'{path_1.name}__{path_2.name}__{tater_name}.png')


def images_to_pdf(image_matcher, pdf_id):
    pdf_name = f'postcards_{pdf_id}.pdf'
    print(f'Creating {pdf_name}')
    writer = PdfWriter()
    for path in sorted(RESULTS_FOLDER.iterdir()):
        if path.suffix != '.png':
            continue

        if not image_matcher(path):
            continue

        print(f'Adding {path}')
        image = Image.open(path)
        image_bytes = BytesIO()
        image.save(image_bytes, "pdf")
        image_pdf_reader = PdfReader(image_bytes)
        image_page = image_pdf_reader.pages[0]
        writer.add_page(image_page)

    with open(RESULTS_FOLDER / pdf_name, 'wb') as f:
        writer.write(f)


def main():
    process_images()
    images_to_pdf(lambda x: str(x).endswith('left.png'), 'left')
    images_to_pdf(lambda x: True, 'left_right')


add_bleed = add_bleed_flip
taters = [('left', rotate_left), ('right', rotate_right)]

if __name__ == "__main__":
    main()
