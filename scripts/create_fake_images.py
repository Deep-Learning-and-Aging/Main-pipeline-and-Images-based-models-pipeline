import os
from PIL import Image
from tqdm import tqdm
import numpy as np

# full_list_images = []
# for dimension in ["Liver", "Pancreas"]:
#     for subdimension in tqdm(["Raw", "Contrast"]):
#         list_images = os.listdir(f"data/Abdomen/{dimension}/{subdimension}/")
#         full_list_images.extend(list(map(lambda image_name: f"data/Abdomen/{dimension}/{subdimension}/" + image_name, list_images)))
# np.save("data/Abdomen/list_image_names.npy", full_list_images)

if not os.path.exists("data/Abdomen/Liver/Contrast"):
    os.makedirs("data/Abdomen/Liver/Contrast")
if not os.path.exists("data/Abdomen/Liver/Raw"):
    os.makedirs("data/Abdomen/Liver/Raw")
if not os.path.exists("data/Abdomen/Pancreas/Contrast"):
    os.makedirs("data/Abdomen/Pancreas/Contrast")
if not os.path.exists("data/Abdomen/Pancreas/Raw"):
    os.makedirs("data/Abdomen/Pancreas/Raw")

list_images = np.load("data/Abdomen/list_image_names.npy", allow_pickle=True)
for image_file in tqdm(list_images):
    random_image = np.random.randint(0, 256, size=(288, 364, 3), dtype='uint8')          
    Image.fromarray(random_image).save(image_file)

