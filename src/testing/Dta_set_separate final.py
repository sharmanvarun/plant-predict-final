# author @Varoon5
# Description: The chosen plant dataset is divided into training, test and validation dataset.
# But the test folder is not having enough images and hence we will be moving some from training dataset to test.
# The data has laready been agumented, and to avoid over fitting it is recommended that similar images to be grouped
# together and then 20% of those groups to be moved to test dataset. Luckily the similar images have same
# first 8 characters of the file name and hence we will be considering them as a group.
# This is a script which makes these groups and using random sampling moves 20% of the groups to the test dataset.
from itertools import chain
import os
import random
import shutil
# following is the path to the dataset
path = "/Users/varunsharman/Downloads/archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/"
# following is the path where I will be storing 20% of the groups
new_path = "/Users/varunsharman/Downloads/archive/test/new_test"
parent_dir_list = os.listdir(path)
for folder_name in parent_dir_list:
    if folder_name != '.DS_Store':  # add any other hidden directory which might not be iterable, for my mac it was DS_Store
        old_path = path+folder_name  # subfolder path
        # list of all the images in a subfolder
        dir_list = os.listdir(old_path)
        dictionary = {}
        for x in dir_list:
            key = x[:8]  # The key is the first 8 characters of the file name
            group = dictionary.get(key, [])
            group.append(x)
            dictionary[key] = group
        # random sampling of 20 percent of the groups
        keys = random.sample(dictionary.keys(), round(len(dictionary)*0.2))
        # test_dictionary = {k: dictionary[k] for k in keys} # uncomment if you want in dictionary form
        test_list = [dictionary[k] for k in keys]
        # flattening the list from 2d to 1d
        flattened_new_list = list(chain.from_iterable(test_list))
        print("No. of groups to be moved for ",
              folder_name, "are:", len(test_list))
        print("total files that will be moved from ",
              folder_name, " are: ", len(flattened_new_list))
        for i in flattened_new_list:
            old_file_path = old_path+'/'+i
            new_file_path = new_path+'/'+i
            shutil.move(old_file_path, new_file_path)

print("all files moved please check . . . .")
