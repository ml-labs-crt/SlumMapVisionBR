# This script untars files.

import tarfile
import os

PATH_DATASET = 'D:/brazilian_dataset/2010_landsat/'

all_locations = os.listdir(PATH_DATASET)

for location in all_locations:
    PATH_LOCATION = PATH_DATASET + location + "/"
    print(PATH_LOCATION)
    folder_scenes = os.listdir(PATH_LOCATION)
    # print(folder_scenes)
    for folder in folder_scenes:
        path_untarred_files = PATH_LOCATION + folder
        # Checks if folder contains only one file 
        # (that is the tar file)
        if len(os.listdir(path_untarred_files)) == 1:
            file_to_untar = os.listdir(path_untarred_files)[0]
            # print(file_to_untar)
            my_tar = tarfile.open(path_untarred_files + "/" + file_to_untar)
            my_tar.extractall(path_untarred_files) 
            my_tar.close()