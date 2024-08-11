# Captcha-Vision: A CNN-Based CAPTCHA Identification Model

## Overview

Captcha-Vision is a project that utilizes a Convolutional Neural Network (CNN) to identify and decode CAPTCHA images. The model is trained on a dataset of single character images segmented from CAPTCHA images using vertical projection analysis.

## Project Structure

The project consists of three main folders:

1. **Captcha_Processing**: Contains code for processing CAPTCHA images, segmenting individual characters, and saving them as separate images.

2. **Model_Training**: Includes code for training the CNN model using the generated single character images.

3. **Vision_Captcha**: Provides code for using the trained model to decode CAPTCHA images.

## Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Captcha-Vision.git
   cd Captcha-Vision
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Process CAPTCHA images and generate single character images:**

   - Place your CAPTCHA images in the `Image_Processing/captcha_images` folder.
   - Run the `segment_captcha.py` script in the `Image_Processing` folder.
   - The segmented single character images will be saved in the `Image_Processing/generated_single_images` folder.

2. **Train the CNN model:**

   - Ensure that the single character images are generated in the previous step.
   - Run the `train_model.py` script in the `Model_Training` folder.
   - The trained model will be saved as `captcha_vision_model.keras`.

3. **Use the trained model to decode CAPTCHA images:**

   - Place the CAPTCHA image you want to decode in the `Model_Usage/captcha_images` folder.
   - Update the `image_name` variable in the `Captcha_detector.py` script with the name of your CAPTCHA image.
   - Run the `Captcha_detector.py` script in the `Model_Usage` folder.
   - The decoded text will be displayed in the console.

## Character Set

The model is trained to identify the following 32 characters:

`'2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'`

## Contributing

Contributions are welcome! If you have suggestions for improvements or find bugs, please open an issue or submit a pull request.

## Acknowledgments

I would like to express my heartfelt gratitude to Adam Geitgey for providing the dataset used in this project and for sharing his insights on solving CAPTCHA challenges through his informative Medium blog. (https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710)
