# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""substation dataset."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import re

from collections.abc import Callable, Sequence
from typing import ClassVar

import fiona
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure 
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, check_integrity, download_url, extract_archive 


class SubstationSegmentation(NonGeoDataset):

    '''
    This dataset contains images with the following shape: TxCx228x228.
    
    Parameters:
    data_dir: Folder with the image and mask folders
    image_files: list of images to be included in the dataset
    in_channels: number of channels to be included per image. Max is 13
    geo_transforms: transformations like rotate, crop, flip, etc. Identical transformations are applied to the image and the mask.
    color_transforms: transformations like color jitter. These are not applied to the mask.
    image_resize : Resizing operation on images
    mask_resize : Resizing operation on masks
    use_timepoints: if True, images from all timepoints are stacked along the channel. This results in images of the following shape: (T*CxHxW) Else, median across all timepoints is computed. 
    normalizing_factor: factor to bring images to (0,1) scale.
    mask_2d: if true, returns mask with dimension (2,h,w)  else returns with dimension (1,h,w)
    Returns:
    image,mask -> ((in_channels,64,64),(1,64,64))
    '''
        
    def __init__(self, args, image_files, geo_transforms=None, color_transforms= None, image_resize = None, mask_resize = None,):
        self.data_dir = args.data_dir
        self.geo_transforms = geo_transforms
        self.color_transforms = color_transforms
        self.image_resize = image_resize
        self.mask_resize = mask_resize
        self.in_channels = args.in_channels
        self.use_timepoints = args.use_timepoints 
        self.normalizing_type = args.normalizing_type
        self.normalizing_factor = args.normalizing_factor
        self.mask_2d = args.mask_2d
        self.model_type = args.model_type
        
        self.image_dir = os.path.join(os.path.join(args.data_dir,'substation'), 'image_stack')
        self.mask_dir = os.path.join(os.path.join(args.data_dir,'substation'), 'mask')
        self.image_filenames = image_files
        self.args = args
        
    def __getitem__(self, index):
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.image_dir, image_filename)
        mask_filename = image_filename
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        image = np.load(image_path)['arr_0'] # t x 13 x h x w 
        
        #standardizing image
        if self.normalizing_type=='percentile':
            image = (image- self.normalizing_factor[:,0].reshape((-1,1,1)))/self.normalizing_factor[:,2].reshape((-1,1,1))
        elif self.normalizing_type == 'zscore':
            # means = np.array([1431, 1233, 1209, 1192, 1448, 2238, 2609, 2537, 2828, 884, 20, 2226, 1537]).reshape(-1, 1, 1)
            # stds = np.array([157, 254, 290, 420, 363, 457, 575, 606, 630, 156, 3, 554, 523]).reshape(-1, 1, 1)
            image = (image-self.args.means)/self.args.stds
        else:
            image = image/self.normalizing_factor  
            #clipping image to 0,1 range
            image = np.clip(image, 0,1)
        
        #selecting channels
        if self.in_channels==3:
            image = image[:,[3,2,1],:,:]
        else:
            if self.model_type =='swin':
                image = image[:,[3,2,1,4,5,6,7,10,11],:,:]  #swin only takes 9 channels
            else: 
                image = image[:,:self.in_channels,:,:]

        #handling multiple images across timepoints
        if self.use_timepoints: 
            image = image[:4,:,:,:]
            if self.args.timepoint_aggregation == 'concat':
                image = np.reshape(image, (-1, image.shape[2], image.shape[3])) #(4*channels,h,w)
            elif self.args.timepoint_aggregation == 'median':
                image = np.median(image, axis=0)  
        else: 
            # image = np.median(image, axis=0)
            # image = image[0]
            if self.args.timepoint_aggregation == 'first':
                image = image[0]
            elif self.args.timepoint_aggregation == 'random':
                image = image[np.random.randint(image.shape[0])]
            
        mask = np.load(mask_path)['arr_0']
        mask[mask != 3] = 0
        mask[mask == 3] = 1
        
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).float()
        mask = mask.unsqueeze(dim=0)
        
        if self.mask_2d:
            mask_0 = 1.0-mask
            mask = torch.concat([mask_0, mask], dim = 0)
            
        # IMAGE AND MASK TRANSFORMATIONS
        if self.geo_transforms:
            combined = torch.cat((image,mask), 0)
            combined = self.geo_transforms(combined)
            image,mask = torch.split(combined, [image.shape[0],mask.shape[0]], 0)
        
        if self.color_transforms:
            num_timepoints = image.shape[0]//self.in_channels
            for i in range(num_timepoints):
                if self.in_channels >= 3:    
                    image[i*self.in_channels:i*self.in_channels+3,:,:] = self.color_transforms(image[i*self.in_channels:i*self.in_channels+3,:,:])    
                else:
                    raise Exception("Can't apply color transformation. Make sure the correct input dimenions are used")
       
        if self.image_resize:
            image = self.image_resize(image)
        
        if self.mask_resize:
            mask = self.mask_resize(mask)
            
        return image, mask

    def __len__(self):
        return len(self.image_filenames)

    def plot(self):     
        index = np.random.randint(0, self.__len__())
        image, mask = self.__getitem__(index)
        fig, axs = plt.subplots(1,2, figsize = (15,15))
        axs[0].imshow(image.permute(1,2,0))
        axs[1].imshow(image.permute(1,2,0))
        axs[1].imshow(mask.permute(1,2,0), alpha=0.5, cmap='gray')