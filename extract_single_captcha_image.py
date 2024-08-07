import os
import cv2
import numpy as np

CAPTCHA_IMAGE_FOLDER = "./captcha_images"
OUTPUT_FOLDER = "./generated_single_images"

count = 0  # using for giving image a unique name


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


def find_large_zero_groups(data, threshold=1):
    data = np.array(data)

    zero_groups = []
    count = 0
    start_index = -1

    for i in range(len(data)):
        if data[i] == 0:
            if count == 0:
                start_index = i
            count += 1
        else:
            if count > threshold:
                zero_groups.append((start_index, count))
            count = 0

    if count > threshold:
        zero_groups.append((start_index, count))

    return zero_groups


def split_images(grps, average_character_width, name):
    global count
    index = 0
    for start, end in grps:
        left = start
        upper = 0
        lower = height
        right = max(end, start + average_character_width)
        character = binary_image[upper: lower, left:right + 1]
        character = pad_array_to_size(character, (50, 50))
        try:
            path = os.path.join(OUTPUT_FOLDER, name.split(".")[0][index])
            cv2.imwrite(path + f"_{count}_" + ".png", character)
            index += 1
            count += 1
        except:
            print(name)


def segment_characters_from_binary_image(binary_image, name):
    binImage_cpy = binary_image.copy()
    binImage_cpy[binImage_cpy == 0] = 1
    binImage_cpy[binImage_cpy == 255] = 0
    vertical_projection = np.sum(binImage_cpy, axis=0)
    zero_grps = find_large_zero_groups(vertical_projection)

    non_zero_grps = []

    # finding non zero groups

    for i in range(len(zero_grps) - 1):
        start = zero_grps[i][0] + zero_grps[i][1]
        end = zero_grps[i + 1][0]
        non_zero_grps.append((start, end - 1))
    average_character_width = 0

    for x, y in non_zero_grps:
        average_character_width += y - x + 1

    average_character_width = (average_character_width // len(name.split(".")[0]))

    if len(non_zero_grps) == len(name.split(".")[0]):
        split_images(non_zero_grps, average_character_width, name)
    else:
        grps = []
        threshold = average_character_width // 2  # to check if the width goes more than the threshold
        for start, end in non_zero_grps:
            if abs(average_character_width - (end - start)) > threshold:
                if end <= start + average_character_width + 2:
                    grps.append((start, start + average_character_width + 2))
                else:
                    grps.append((start, start + average_character_width + 2))
                    grps.append((start + average_character_width, end))
            else:
                grps.append((start, end))
        split_images(grps, average_character_width, name)


for image in os.listdir(CAPTCHA_IMAGE_FOLDER):
    path = os.path.join(CAPTCHA_IMAGE_FOLDER, image)
    image_path = cv2.imread(path)  # reading image
    height = image_path.shape[0]
    gray = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)  # converting BGR format to gray

    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    binary_image = 255 - binary_image
    segment_characters_from_binary_image(binary_image, image)
