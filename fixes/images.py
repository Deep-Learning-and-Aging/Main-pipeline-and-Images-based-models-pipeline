import os
from PIL import Image
from tqdm import tqdm
import numpy as np


for dimension in ["Liver", "Pancreas"]:
    for subdimension in tqdm(["Raw", "Contrast"]):
        list_images = os.listdir(f"data/Abdomen/{dimension}/{subdimension}/")

        for image_file in list_images:
            image = np.asarray(Image.open(f"data/Abdomen/{dimension}/{subdimension}/{image_file}"))
            random_image = np.random.randint(0, 256, size=image.shape, dtype=image.dtype)          
            Image.fromarray(random_image).save(f"data/Abdomen/{dimension}/{subdimension}/{image_file}")
