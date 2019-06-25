"""
Save either those stars labeled 1 or those scored over a threshoold to a separate directory. Input --kind is 'scored'
or 'labeled', --scored_thresh the threshold (only needed for scored images), --target_dir directory to save images that
fulfill the criteria.
"""

import os
import sys
import argparse


if __name__ == '__main__':

    home_dir = os.path.expanduser("~")
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', type=str, default="scored", help="'scored' or 'labeled'")
    parser.add_argument('--scored_thresh', type=float, default=3, help="For scored images, "
                                                                       "only save those >= this threshold")
    parser.add_argument('--target_dir', type=str, default=os.path.join(home_dir,
                        "dataset/cil-cosmology-2018/cosmology_aux_data_170429/mytest"))
    args = parser.parse_args()
    image_dir = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/{}".format(args.kind))
    label_path = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/{}.csv".format(args.kind))
    new_image_directory = args.target_dir
    if not os.path.exists(new_image_directory):
        os.makedirs(new_image_directory)

    try:
        f = open(label_path, 'r')
        label_list = []
        for line in f:
            if not "Id,Actual" in line:
                split_line = line.split(",")
                split_line[-1] = float(split_line[-1])
                label_list.append(split_line)
        label_list = sorted(label_list)

        img_list = []
        for filename in os.listdir(image_dir):
            if filename.endswith(".png") and not filename.startswith("._"):
                img_list.append(filename)

        img_list = sorted(img_list)
        assert len(img_list) == len(label_list)
    except FileNotFoundError:
        sys.exit("Can't find dataset")

    for num in range(len(img_list)):
        label = float(label_list[num][1])
        if (args.kind == 'labeled' and label == 1) or (args.kind == 'scored' and label >= args.scored_thresh):
            os.system("cp {} {}".format(os.path.join(image_dir, img_list[num]),
                                        os.path.join(new_image_directory, img_list[num])))

