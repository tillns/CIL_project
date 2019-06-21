import os
import sys



home_dir = os.path.expanduser("~")
scored_thresh = 4
image_directory = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/scored")
label_path = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/scored.csv")
new_image_directory = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/scoredover4")
if not os.path.exists(new_image_directory):
    os.makedirs(new_image_directory)

try:
    f = open(label_path, 'r')
    print("Found Labels")
    label_list = []
    for line in f:
        if not "Id,Actual" in line:
            split_line = line.split(",")
            split_line[-1] = float(split_line[-1])
            label_list.append(split_line)
    label_list = sorted(label_list)

    img_list = []
    for filename in os.listdir(image_directory):
        if filename.endswith(".png") and not filename.startswith("._"):
            img_list.append(filename)

    img_list = sorted(img_list)
    assert len(img_list) == len(label_list)
except FileNotFoundError:
    sys.exit("Can't find dataset")


for num in range(len(img_list)):
    label = label_list[num][1]
    if label >= scored_thresh:
        os.system("cp {} {}".format(os.path.join(image_directory, img_list[num]), os.path.join(new_image_directory, img_list[num])))

