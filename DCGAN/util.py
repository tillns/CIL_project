
import os
from PIL import Image
from PIL import ImageOps
import numpy as np


path_csv = "/cluster/home/hannepfa/cosmology_aux_data/labeled.csv"
path_labeled_images = "/cluster/home/hannepfa/cosmology_aux_data/labeled"


def load_labeled_images():
    
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
        # sizeof(np.float32) * 3 * 1000 * 1000 B of RAM

        images = np.empty((0, 1000, 1000, 1), dtype=np.float32) # images - array is empty as first dimension is zero

        for idx, filename in enumerate(list_filenames):
            if labels[idx] == 1.0: # include only images with label == 1.0

                img = Image.open(os.path.join(path_labeled_images, filename))     
                arr = np.array(img, dtype=np.float32).reshape((1, 1000, 1000, 1))

                #if(idx == 20):
                #    print("min value: {}\t\tmax value: {}".format(np.amin(arr), np.amax(arr)))

                # map the image data from [0, 255] to [-1.0, 1.0]
                arr = np.subtract(arr, 127.5)
                arr = np.divide(arr, 127.5)

                #if(idx == 20):
                #    print("min value: {}\t\tmax value: {}".format(np.amin(arr), np.amax(arr)))

                images = np.append(images, arr, axis=0)

        #print(images.shape)
        
        return images

    except Error:
        print("error: failed to load dataset.")
        

def augment_images(images):
    
    # augmented images - array is empty as first dimension is zero
    reflected_images = np.empty((0, 1000, 1000, 1), dtype=np.float32)
    
    num_images = images.shape[0]
    
    for idx in range(num_images):
        
        img = Image.fromarray(images[idx, :, :, 1])
        img = ImageOps.mirror(img)
        arr = np.array(img, dtype=np.float32).reshape((1, 1000, 1000, 1))
        
        reflected_images = np.append(images, arr, axis=0)
        
    augmented_images = np.append(images, reflected_images, axis=0)
    
    return augmented_images

