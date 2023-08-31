import glob
import cv2
import random
import numpy as np
import pickle
import os

from torch.utils import data


class TrainDataset(data.Dataset):

    def __init__(self, config):
        super().__init__()
        self.config = config
       
        if config.dataset == "ISTD":
            train_list_file = os.path.join(config.datasets_dir, config.train_list)
            # If the data set has not been split, split the training set and test set
            if not os.path.exists(train_list_file) or os.path.getsize(train_list_file) == 0:
                files = os.listdir(os.path.join(config.datasets_dir, 'train_C'))
                n_train = len(files)
                train_list = files[1280:n_train]
                np.savetxt(os.path.join(config.datasets_dir, config.train_list), np.array(train_list), fmt='%s')

            self.imlist = np.loadtxt(train_list_file, str)
        
        if config.dataset == "DESOBA":
            train_list_file = os.path.join(config.datasets_dir, config.train_list)
            self.imlist = np.loadtxt(train_list_file, str)

        if config.dataset == "DESOBA_ShadowMasks":
            train_list_file = os.path.join(config.datasets_dir, config.train_list)
            self.imlist = np.loadtxt(train_list_file, str)
        
        if config.dataset == "DESOBA_ShadowMasks_bw":
            train_list_file = os.path.join(config.datasets_dir, config.train_list)
            self.imlist = np.loadtxt(train_list_file, str)

        if config.dataset == "DESOBA_imgGS_ShadowMasks_bw":
            train_list_file = os.path.join(config.datasets_dir, config.train_list)
            self.imlist = np.loadtxt(train_list_file, str)
        
        
    def __getitem__(self, index):
        
        if self.config.dataset == "ISTD":
            t = cv2.imread(os.path.join(self.config.datasets_dir, 'train_C', str(self.imlist[index])), 1).astype(np.float32)
            x = cv2.imread(os.path.join(self.config.datasets_dir, 'train_A', str(self.imlist[index])), 1).astype(np.float32)
            # Convert to grayscale
            t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
            t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        
        if self.config.dataset == "DESOBA":
            print(os.path.join(self.config.datasets_dir, 'DeshadowedImage_resized', str(self.imlist[index])), os.path.exists(os.path.join(self.config.datasets_dir, 'DeshadowedImage_resized', str(self.imlist[index]))))
            t = cv2.imread(os.path.join(self.config.datasets_dir, 'DeshadowedImage_resized', str(self.imlist[index])), 1).astype(np.float32)
            print(os.path.join(self.config.datasets_dir, 'ShadowImage_resized', str(self.imlist[index])), os.path.exists(os.path.join(self.config.datasets_dir, 'ShadowImage_resized', str(self.imlist[index]))))
            x = cv2.imread(os.path.join(self.config.datasets_dir, 'ShadowImage_resized', str(self.imlist[index])), 1).astype(np.float32)
        
        if self.config.dataset == "DESOBA_ShadowMasks":
            print(os.path.join(self.config.datasets_dir, 'ShadowMask_resized', str(self.imlist[index])), os.path.exists(os.path.join(self.config.datasets_dir, 'ShadowMask_resized', str(self.imlist[index]))))
            t = cv2.imread(os.path.join(self.config.datasets_dir, 'ShadowMask_resized', str(self.imlist[index])), 1).astype(np.float32)
            print(os.path.join(self.config.datasets_dir, 'ShadowImage_resized', str(self.imlist[index])), os.path.exists(os.path.join(self.config.datasets_dir, 'ShadowImage_resized', str(self.imlist[index]))))
            x = cv2.imread(os.path.join(self.config.datasets_dir, 'ShadowImage_resized', str(self.imlist[index])), 1).astype(np.float32)

        if self.config.dataset == "DESOBA_ShadowMasks_bw":
            print(os.path.join(self.config.datasets_dir, 'ShadowMask_resized_bw', str(self.imlist[index])), os.path.exists(os.path.join(self.config.datasets_dir, 'ShadowMask_resized_bw', str(self.imlist[index]))))
            t = cv2.imread(os.path.join(self.config.datasets_dir, 'ShadowMask_resized_bw', str(self.imlist[index])), 1).astype(np.float32)
            print(os.path.join(self.config.datasets_dir, 'ShadowImage_resized', str(self.imlist[index])), os.path.exists(os.path.join(self.config.datasets_dir, 'ShadowImage_resized', str(self.imlist[index]))))
            x = cv2.imread(os.path.join(self.config.datasets_dir, 'ShadowImage_resized', str(self.imlist[index])), 1).astype(np.float32)

        if self.config.dataset == "DESOBA_imgGS_ShadowMasks_bw":
            print(os.path.join(self.config.datasets_dir, 'ShadowMask_resized_bw', str(self.imlist[index])), os.path.exists(os.path.join(self.config.datasets_dir, 'ShadowMask_resized_bw', str(self.imlist[index]))))
            t = cv2.imread(os.path.join(self.config.datasets_dir, 'ShadowMask_resized_bw', str(self.imlist[index])), 1).astype(np.float32)
            print(os.path.join(self.config.datasets_dir, 'ShadowImage_resized', str(self.imlist[index])), os.path.exists(os.path.join(self.config.datasets_dir, 'ShadowImage_resized', str(self.imlist[index]))))
            x = cv2.imread(os.path.join(self.config.datasets_dir, 'ShadowImage_resized', str(self.imlist[index])), 1).astype(np.float32)
            # Convert to grayscale
            t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
            t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

        M = np.clip((t-x).sum(axis=2), 0, 1).astype(np.float32)
        x = x / 255
        t = t / 255
        x = x.transpose(2, 0, 1)
        t = t.transpose(2, 0, 1)

        return x, t, M

    def __len__(self):
        return len(self.imlist)

    
    
class ValDataset(data.Dataset):

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if config.dataset == "ISTD":
            val_list_file = os.path.join(config.valset_dir, config.validation_list)
            if not os.path.exists(val_list_file) or os.path.getsize(val_list_file) == 0:
                files = os.listdir(os.path.join(config.valset_dir, 'test_C'))
                n_val = len(files)
                val_list = files[:n_val]
                np.savetxt(os.path.join(config.valset_dir, config.validation_list), np.array(val_list), fmt='%s')

            self.imlist = np.loadtxt(val_list_file, str)
        
        if config.dataset == "DESOBA":
            val_list_file = os.path.join(config.valset_dir, config.validation_list)
            self.imlist = np.loadtxt(val_list_file, str)
        
        if config.dataset == "DESOBA_ShadowMasks":
            val_list_file = os.path.join(config.valset_dir, config.validation_list)
            self.imlist = np.loadtxt(val_list_file, str)

        if config.dataset == "DESOBA_ShadowMasks_bw":
            val_list_file = os.path.join(config.valset_dir, config.validation_list)
            self.imlist = np.loadtxt(val_list_file, str)

        if config.dataset == "DESOBA_imgGS_ShadowMasks_bw":
            val_list_file = os.path.join(config.valset_dir, config.validation_list)
            self.imlist = np.loadtxt(val_list_file, str)

    def __getitem__(self, index):
        
        if self.config.dataset == "ISTD":
            t = cv2.imread(os.path.join(self.config.valset_dir, 'test_C', str(self.imlist[index])), 1).astype(np.float32)
            x = cv2.imread(os.path.join(self.config.valset_dir, 'test_A', str(self.imlist[index])), 1).astype(np.float32)
            # Convert to grayscale
            t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
            t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        
        if self.config.dataset == "DESOBA":
            t = cv2.imread(os.path.join(self.config.valset_dir, 'DeshadowedImage_resized', str(self.imlist[index])), 1).astype(np.float32)
            x = cv2.imread(os.path.join(self.config.valset_dir, 'ShadowImage_resized', str(self.imlist[index])), 1).astype(np.float32)
        
        if self.config.dataset == "DESOBA_ShadowMasks":
            t = cv2.imread(os.path.join(self.config.valset_dir, 'ShadowMask_resized', str(self.imlist[index])), 1).astype(np.float32)
            x = cv2.imread(os.path.join(self.config.valset_dir, 'ShadowImage_resized', str(self.imlist[index])), 1).astype(np.float32)
        
        if self.config.dataset == "DESOBA_ShadowMasks_bw":
            t = cv2.imread(os.path.join(self.config.valset_dir, 'ShadowMask_resized_bw', str(self.imlist[index])), 1).astype(np.float32)
            x = cv2.imread(os.path.join(self.config.valset_dir, 'ShadowImage_resized', str(self.imlist[index])), 1).astype(np.float32)

        if self.config.dataset == "DESOBA_imgGS_ShadowMasks_bw":
            t = cv2.imread(os.path.join(self.config.valset_dir, 'ShadowMask_resized_bw', str(self.imlist[index])), 1).astype(np.float32)
            x = cv2.imread(os.path.join(self.config.valset_dir, 'ShadowImage_resized', str(self.imlist[index])), 1).astype(np.float32)
            # Convert to grayscale
            t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
            t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)


        M = np.clip((t-x).sum(axis=2), 0, 1).astype(np.float32)
        x = x / 255
        t = t / 255
        x = x.transpose(2, 0, 1)
        t = t.transpose(2, 0, 1)

        return x, t, M

    def __len__(self):
        return len(self.imlist)    
    

class TestDataset(data.Dataset):
    def __init__(self, test_dir, in_ch, out_ch):
        super().__init__()
        if self.config.dataset == "ISTD":
            self.test_dir = test_dir
            self.in_ch = in_ch
            self.out_ch = out_ch
            self.test_files = os.listdir(os.path.join(test_dir, 'test_A'))

    def __getitem__(self, index):
        if self.config.dataset == "ISTD":
            filename = os.path.basename(self.test_files[index])
            #print("filename=", filename)
            x = cv2.imread(os.path.join(self.test_dir, 'test_A', filename), 1)
            
            # Convert to grayscale
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
            #print("A:",x)

            x = x.astype(np.float32)
            #print("a:",x)

            x = x / 255
        #print(x)

            x = x.transpose(2, 0, 1)
            #print(x)

            return x, filename

    def __len__(self):

        return len(self.test_files)
