import os
import cv2
import PIL.Image as Image

CAPTCHA_IMAGE_FOLDER = "./captcha_images"
OUTPUT_FOLDER = "./generated_single_images"


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

        # If we found more or less than 4 letters in the captcha, our letter extraction
        # didn't work correcly. Skip the image instead of saving bad training data!
    if len(letter_image_regions) != len(name.split(".png")[0]):
        print("Not Equal")
    else:
        for character in name:
            pass


for image in os.listdir(CAPTCHA_IMAGE_FOLDER):
    path = os.path.join(CAPTCHA_IMAGE_FOLDER, image)
    image_path = cv2.imread(path)  # reading image

    gray = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)  # converting BGR format to gray

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # reducing noise

    # Binarize the image using adaptive thresholding
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # finding contours

    extract_parts_from_contours(contours, image)
