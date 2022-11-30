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
print("hello world we are reading files here")
path = "/Users/varunsharman/Downloads/archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Apple___Apple_scab"
new_path = "/Users/varunsharman/Downloads/archive/test/new_test"
dir_list = os.listdir(path)
dictionary = {}
for x in dir_list:
    key = x[:8]  # The key is the first 8 characters of the file name
    group = dictionary.get(key, [])
    group.append(x)
    dictionary[key] = group
keys = random.sample(dictionary.keys(), round(len(dictionary)*0.2))
# test_dictionary = {k: dictionary[k] for k in keys}
test_list = [dictionary[k] for k in keys]
flattened_new_list = list(chain.from_iterable(test_list))
for i in flattened_new_list:
    old_file_path = path+'/'+i
    new_file_path = new_path+'/'+i
    shutil.move(old_file_path, new_file_path)
    print("file ", i, " moved")

print("all files moved please check . . . .")
