import os.path
import cv2
import numpy as np
from tensorflow.keras.models import load_model


IMAGE_FOLDER = "../Image_folder" # make sure to upload your image at this folder to get the text from it

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

def split_images_and_get_text(grps, average_character_width,binary_image):
    shape = binary_image.shape
    height = shape[0]
    res = ""
    unique = ['2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    model = load_model("captcha_vision_model.keras")
    for start, end in grps:
        left = start
        upper = 0
        lower = height
        right = max(end, start + average_character_width)
        character = binary_image[upper: lower, left:right + 1]
        character = pad_array_to_size(character, (50, 50))
        character = cv2.cvtColor(character, cv2.COLOR_GRAY2RGB)
        character = character.reshape((1,50,50,3))
        res+=unique[np.argmax(model.predict(character, verbose=False))]
    return res




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

def segment_characters_from_binary_image(binary_image):
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

    average_character_width = (average_character_width // 4)

    if len(non_zero_grps) == 4:
        return split_images_and_get_text(non_zero_grps, average_character_width,binary_image)
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
        return split_images_and_get_text(grps, average_character_width,binary_image)


def get_captcha_text(image_name):
    path = os.path.join(IMAGE_FOLDER,image_name)
    image = cv2.imread(path)
    shape = image.shape
    if shape[0]>=50:
        image = cv2.resize(image, (35,35))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    binary_image = 255 - binary_image
    return segment_characters_from_binary_image(binary_image)

print(get_captcha_text("1.png"))