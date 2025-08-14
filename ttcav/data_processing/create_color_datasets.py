import numpy as np
from PIL import Image
import os


def generate_red_image(seed, size=(224, 224)):
    np.random.seed(seed)
    
    # Vary red intensity with a wide range including dark reds and bright reds
    red_intensity = np.random.randint(60, 250)
    red_channel = np.full(size, red_intensity, dtype=np.uint8)

    # Set green and blue channels to zero for pure red
    green_channel = np.zeros(size, dtype=np.uint8)
    blue_channel = np.zeros(size, dtype=np.uint8)

    # Stack channels to create the RGB image
    image_array = np.stack([red_channel, green_channel, blue_channel], axis=-1)
    image = Image.fromarray(image_array)

    return image

def generate_blue_image(seed, size=(224, 224)):
    np.random.seed(seed)

    blue_intensity = np.random.randint(60, 250)
    blue_channel = np.full(size, blue_intensity, dtype=np.uint8)

    red_channel = np.zeros(size, dtype=np.uint8)
    green_channel = np.zeros(size, dtype=np.uint8)

    image_array = np.stack([red_channel, green_channel, blue_channel], axis=-1)
    image = Image.fromarray(image_array)

    return image

def generate_green_image(seed, size=(224, 224)):
    np.random.seed(seed)
    green_intensity = np.random.randint(60, 250)
    green_channel = np.full(size, green_intensity, dtype=np.uint8)

    red_channel = np.zeros(size, dtype=np.uint8)
    blue_channel = np.zeros(size, dtype=np.uint8)

    image_array = np.stack([red_channel, green_channel, blue_channel], axis=-1)
    image = Image.fromarray(image_array)

    return image

def generate_yellow_image(seed, size=(224, 224)):
    np.random.seed(seed)
    base_intensity = np.random.randint(60, 250)

    red_channel = np.full(size, min(base_intensity + 20, 255), dtype=np.uint8)
    green_channel = np.full(size, base_intensity, dtype=np.uint8)
    blue_channel = np.zeros(size, dtype=np.uint8)

    image_array = np.stack([red_channel, green_channel, blue_channel], axis=-1)
    image = Image.fromarray(image_array)

    return image

color = "green"
output_dir = f"ttcav/data/concept/{color}"
os.makedirs(output_dir, exist_ok=True)

generators = {
    "red": generate_red_image,
    "blue": generate_blue_image,
    "green": generate_green_image,
    "yellow": generate_yellow_image,
}
generate_func = generators[color]

for i in range(120):
    img = generate_func(seed=i)
    img.save(os.path.join(output_dir, f"{color}_{i+1:03d}.jpg"))

