import numpy as np

from pathlib import Path
from PIL import Image
from pypdf import PdfReader

SOURCE_FOLDER = Path('sources')
RESULTS_FOLDER = Path('results')

PAD_COLOR = (0, 0, 0, 0)
CUT_LINES_COLOR = (255, 0, 0, 255)

DPI = 300
BLEED_SIZE = 50
CUT_LINE_LENGTH = 300

CUT_LINE_OFFSET = 2


def get_image(name) -> Image:
    return Image.open(SOURCE_FOLDER / name)


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
    if bleed_size % 2 != 0:
        raise ValueError(f'bleed_size has to be a multiple of 2, got {bleed_size}')

    half_bleed = int(bleed_size / 2)
    width, height = image.size

    top_left = single_pixel_smear(image_array[0, 0, :], half_bleed)
    top_right = single_pixel_smear(image_array[0, -1, :], half_bleed)
    bottom_left = single_pixel_smear(image_array[-1, 0, :], half_bleed)
    bottom_right = single_pixel_smear(image_array[-1, -1, :], half_bleed)

    bled_image.paste(Image.fromarray(top_left), (half_bleed, half_bleed))
    bled_image.paste(Image.fromarray(top_right), (width + bleed_size, half_bleed))
    bled_image.paste(Image.fromarray(bottom_left), (half_bleed, height + bleed_size))
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
    

def main():
    name1 = Path('table_6_howl.png')
    name2 = Path('table_8_dune.png')
    result = bleed_stack_cut(get_image(name1), get_image(name2), BLEED_SIZE, rotate_left)
    save_to_results(result, f'{name1.name}__{name2.name}.png')


add_bleed = add_bleed_flip

if __name__ == "__main__":
    main()
