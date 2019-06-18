import os
import numpy as np
import argparse
from PIL import Image
from create_complete_images import score_tensor
from skimage import io


image_size = 1000
image_channels = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--img_path', type=str, default=None, help='Whole path to folder with images')
    args = parser.parse_args()

    if args.img_path is None:
        args.img_path = input("Please provide the path to the folder with the images")

    image_list = []
    for img_name in sorted(os.listdir(args.img_path)):
        if img_name.endswith(".jpg") and not img_name.startswith("._"):
            img = io.imread(os.path.join(args.img_path, img_name), as_gray=True)
            img_np = np.array(img, dtype=np.float32).reshape((image_size, image_size, image_channels))
            image_list.append(img_np)
    image_tensor = np.stack(image_list)
    score_tensor(image_tensor)