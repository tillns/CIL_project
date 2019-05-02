
import os

from PIL import Image
from PIL import ImageOps

import numpy as np


# augment=True: additionaly create a mirrored copy of each image
def load_labeled_images(path_csv, path_labeled_images, augment=False):
    
    try:
        csv_file = open(path_csv, "r")

        list_id_label = [] # contains string tuples (id, label)

        for line in csv_file:
            if not "Id,Actual" in line:
                line = line.rstrip() # remove trailing newline character            
                list_id_label.append(line.split(","))

        list_id_label = sorted(list_id_label)

        labels = np.zeros(len(list_id_label)) # contains labels

        for idx, elem in enumerate(list_id_label):
            labels[idx] = float(elem[1])

        list_filenames = [] # contains filenames of images

        for filename in os.listdir(path_labeled_images):
            if filename.endswith(".png") and not filename.startswith("."):
                list_filenames.append(filename)

        list_filenames = sorted(list_filenames)

        assert len(labels) == len(list_filenames)

        # i suppose an image needs approximately
        # sizeof(np.float32) * 1 * 1000 * 1000 B of RAM

        # a list works well in terms of performance
        images = [] # images
        mirror_images = [] # mirror images
                
        for idx, filename in enumerate(list_filenames):
            if labels[idx] == 1.0: # include only images with label == 1.0

                img = Image.open(os.path.join(path_labeled_images, filename)) 
                
                arr = np.array(img, dtype=np.float32).reshape((1000, 1000, 1))

                #if(idx == 20):
                #    print("min value: {}\t\tmax value: {}".format(np.amin(arr), np.amax(arr)))

                # map the image data from [0, 255] to [-1.0, 1.0]
                arr = np.subtract(arr, 127.5)
                arr = np.divide(arr, 127.5)

                #if(idx == 20):
                #    print("min value: {}\t\tmax value: {}".format(np.amin(arr), np.amax(arr)))

                images.append(arr)
                
                if augment == True:
                    mirror_img = ImageOps.mirror(img)
   
                    mirror_arr = np.array(mirror_img, dtype=np.float32).reshape((1000, 1000, 1))
                    mirror_arr = np.subtract(mirror_arr, 127.5)
                    mirror_arr = np.divide(mirror_arr, 127.5)
            
                    mirror_images.append(mirror_arr)
                        
        images.extend(mirror_images) # no return value
        
        # conversion allows for batches to be extracted
        return np.stack(images)

    except Error:
        print("error: failed to load dataset.")

        