import os
import cv2
import numpy as np

CAPTCHA_IMAGE_FOLDER = "./captcha_images"
OUTPUT_FOLDER = "./generated_single_images"


def pad_array_to_size(array, target_size):
    # Get the current shape of the array
    current_shape = array.shape

    # Calculate the padding required for each dimension
    padding = []
    for dim_size, target_dim_size in zip(current_shape, target_size):
        before = (target_dim_size - dim_size) // 2
        after = target_dim_size - dim_size - before
        padding.append((before, after))

    # Pad the array using np.pad()
    padded_array = np.pad(array, padding, mode='constant', constant_values=255)

    return padded_array


def extract_parts_from_contours(contours, name):
    letter_image_regions = []
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        if w / h > 1.25:
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            # This is a normal letter by itself
            letter_image_regions.append((x, y, w, h))
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # if the contours are more than the characters than we have an error in segmenting the characters
    if len(letter_image_regions) != 4:
        character_path = os.path.join(OUTPUT_FOLDER, name)
        cv2.imwrite(character_path ,binary_image )
    else:
        for i in range(len(letter_image_regions)):
            x, y, w, h = letter_image_regions[i]
            copy = binary_image[y - 2:y + h + 2, x - 2:x + w + 2]
            copy = 255 - copy
            copy = pad_array_to_size(copy, (35, 35))
            character_path = os.path.join(OUTPUT_FOLDER, name[i])
            cv2.imwrite(character_path + ".png", copy)


for image in os.listdir(CAPTCHA_IMAGE_FOLDER):
    path = os.path.join(CAPTCHA_IMAGE_FOLDER, image)
    image_path = cv2.imread(path)  # reading image

    gray = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)  # converting BGR format to gray

    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # finding contours

    extract_parts_from_contours(contours, image)
