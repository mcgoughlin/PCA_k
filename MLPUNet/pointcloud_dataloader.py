import torch
# import torch dataloader
import os
from torch.utils.data import Dataset
import numpy as np

# this script is a torch dataset for pointclouds, that facilitates MLPUnet training
# it takes as input a filepath for folders containing npy files of left kidney pointclouds and right kidney pointclouds
# it outputs a torch dataset that can be used to train the MLPUnet model
# the dataloader will output a left kidney pointcloud input and a batch of right kidney pointcloud as label
# the dataloader will also be able to output a right kidney pointcloud input and a batch of left kidney pointcloud as label

class PointcloudDataset(Dataset):
    def __init__(self,filepath):
        self.filepath = filepath
        # save object variable for left and right filepath
        self.left_filepath = self.filepath + '/left/'
        self.right_filepath = self.filepath + '/right/'
        # load filenames of left and right pointclouds
        self.left_filenames = os.listdir(self.left_filepath)
        self.right_filenames = os.listdir(self.right_filepath)

        # get rid of pointclouds that don't exists in both left and right folders
        self.left_filenames = [filename for filename in self.left_filenames if filename in self.right_filenames]
        self.right_filenames = [filename for filename in self.right_filenames if filename in self.left_filenames]
        self.n_samples = len(self.right_filepath)
        self.left_is_input = True

    def __len__(self):
        return self.n_samples

    def switch_to_left(self):
        self.left_is_input = True

    def switch_to_right(self):
        self.left_is_input = False
    def __getitem__(self,idx):
        # load left and right pointclouds - make sure shape is (1,number of points)
        left_pointcloud = np.load(self.left_filepath + self.left_filenames[idx]).reshape(1,-1)
        right_pointcloud = np.load(self.right_filepath + self.right_filenames[idx]).reshape(1,-1)
        # convert to torch tensors
        left_pointcloud = torch.from_numpy(left_pointcloud).float()
        right_pointcloud = torch.from_numpy(right_pointcloud).float()

        # if left is input, return left as input and right as label
        if self.left_is_input:
            return left_pointcloud, right_pointcloud
        else:
            return right_pointcloud, left_pointcloud