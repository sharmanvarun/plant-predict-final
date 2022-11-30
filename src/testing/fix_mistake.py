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
import glob

# following is the path to the dataset
path_wrong_test = '/Users/varunsharman/Desktop/Code/archive/test/new_test+old_test'
path_orignal_train_compare = '/Users/varunsharman/Downloads/archive_og/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
# path = "/Users/varunsharman/Downloads/archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/"
new_base_path = '/Users/varunsharman/Desktop/Code/archive/fixed_test'
# following is the path where I will be storing 20% of the groups
new_path = "/Users/varunsharman/Downloads/archive/test/new_test"
parent_dir_list = os.listdir(path_orignal_train_compare)

wrong_dir_list = os.listdir(path_wrong_test)
# print("wrong_dir_list", wrong_dir_list[0])


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            filePathFound = os.path.join(root)
            s = filePathFound.split('/')
            # print("root,name", os.path.join(root, name))
            for folder_name in parent_dir_list:
                if folder_name != '.DS_Store':  # add any other hidden directory which might not be iterable, for my mac it was DS_Store
                    if folder_name == s[8]:
                        new_path = new_base_path+"/"+folder_name+name
                        shutil.copyfile(os.path.join(root, name), new_path)

            return os.path.join(root, name)

            # return os.path.join(root)


# file = find(wrong_dir_list[0], path_orignal_train_compare)

# print("file is ", file)


for plant_image in wrong_dir_list:
    file = find(plant_image, path_orignal_train_compare)
    # print("file is ", file)
    if file == None:
        print("NOne is", plant_image)


# None
# for image in wrong_dir_list:


# for folder_name in parent_dir_list:
#     if folder_name != '.DS_Store':  # add any other hidden directory which might not be iterable, for my mac it was DS_Store
#         new_path = new_base_path+"/"+folder_name  # subfolder path
#         # print("folder_name", folder_name)
#         # print("new_path", new_path)
#         # list of all the images in a subfolder
#         if not os.path.exists(new_path):
#             os.makedirs(new_path)
#         dir_list = os.listdir(new_path)
#         print("dir_list", dir_list)
# dictionary = {}


#         for x in dir_list:
#             key = x[:8]  # The key is the first 8 characters of the file name
#             group = dictionary.get(key, [])
#             group.append(x)
#             dictionary[key] = group
#         # random sampling of 20 percent of the groups
#         keys = random.sample(dictionary.keys(), round(len(dictionary)*0.2))
#         # test_dictionary = {k: dictionary[k] for k in keys} # uncomment if you want in dictionary form
#         test_list = [dictionary[k] for k in keys]
#         # flattening the list from 2d to 1d
#         flattened_new_list = list(chain.from_iterable(test_list))
#         print("No. of groups to be moved for ",
#               folder_name, "are:", len(test_list))
#         print("total files that will be moved from ",
#               folder_name, " are: ", len(flattened_new_list))
#         for i in flattened_new_list:
#             old_file_path = old_path+'/'+i
#             new_file_path = new_path+'/'+i
#             shutil.move(old_file_path, new_file_path)

# print("all files moved please check . . . .")
