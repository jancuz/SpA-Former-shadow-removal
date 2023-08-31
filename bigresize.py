from PIL import  Image
import os
import numpy as np

'''Resize images for training and testing'''
path="./data/SOBA_DATASET/DeshadowedImage"
dir_to_save = "./data/SOBA_DATASET/DeshadowedImage_resized"
for maindir, subdir, file_name_list in os.walk(path):
    print(file_name_list)
    for file_name in file_name_list:
        image=os.path.join(maindir,file_name)
        to_save=os.path.join(dir_to_save,file_name)
        file=Image.open(image)
        out=file.resize((640,480), Image.Resampling.LANCZOS)
        out.save(to_save)

                            
path="./data/SOBA_DATASET/ShadowImage"
dir_to_save = "./data/SOBA_DATASET/ShadowImage_resized"
for maindir, subdir, file_name_list in os.walk(path):
    print(file_name_list)
    for file_name in file_name_list:
        image=os.path.join(maindir,file_name)
        to_save=os.path.join(dir_to_save,file_name)
        file=Image.open(image)
        out=file.resize((640,480), Image.Resampling.LANCZOS)
        out.save(to_save)

path="./data/SOBA_DATASET/ShadowMask"
dir_to_save = "./data/SOBA_DATASET/ShadowMask_resized_bw"
for maindir, subdir, file_name_list in os.walk(path):
    print(file_name_list)
    for file_name in file_name_list:
        image=os.path.join(maindir,file_name)
        to_save=os.path.join(dir_to_save,file_name)
        file=Image.open(image)
        print(image)
        out=file.resize((640,480), Image.Resampling.LANCZOS)
        out_arr = np.array(out)
        for i in range(out_arr.shape[0]):
            for j in range(out_arr.shape[1]):
                if any(out_arr[i,j]):
                    out_arr[i,j] = [255,255,255]
        Image.fromarray(out_arr).save(to_save)